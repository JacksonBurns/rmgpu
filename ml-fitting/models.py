from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from numpy.typing import ArrayLike

from chemprop import featurizers, models, nn as chemprop_nn
from chemprop.featurizers.atom import RIGRAtomFeaturizer
from chemprop.featurizers.bond import RIGRBondFeaturizer
from chemprop.nn.metrics import ChempropMetric, MetricRegistry, LossFunctionRegistry


CHEMELEON_MOL_FEATURIZER = featurizers.SimpleMoleculeMolGraphFeaturizer()
RIGR_RXN_FEATURIZER = featurizers.CondensedGraphOfReactionFeaturizer(
    atom_featurizer=RIGRAtomFeaturizer(), bond_featurizer=RIGRBondFeaturizer()
)


def smooth_clamp(x, min_val, max_val, beta=5.0):
    # approximate clamp using softplus
    # 1. Soft approximation of max(min_val, x)
    low_clip = min_val + F.softplus(x - min_val, beta=beta)
    # 2. Soft approximation of min(max_val, low_clip) -> max_val - max(0, max_val - low_clip)
    return max_val - F.softplus(max_val - low_clip, beta=beta)


@MetricRegistry.register("huber")
@LossFunctionRegistry.register("huber")
class HuberMetric(ChempropMetric):
    def _calc_unreduced_loss(self, preds, targets, *args):
        return F.smooth_l1_loss(preds, targets, reduction="none", beta=1.0)


import torch
import torch.nn.functional as F
from torch import nn
from numpy.typing import ArrayLike


def smooth_clamp(
    x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor, beta: float = 5.0
) -> torch.Tensor:
    """Differentiable clamp using softplus."""
    low_clip = min_val + F.softplus(x - min_val, beta=beta)
    return max_val - F.softplus(max_val - low_clip, beta=beta)


class BoundedOutputTransform(nn.Module):
    def __init__(
        self,
        mean: ArrayLike,
        scale: ArrayLike,
        lower_bounds: ArrayLike,
        upper_bounds: ArrayLike,
        pad: int = 0,
    ):
        super().__init__()

        mean_t = torch.as_tensor(mean, dtype=torch.float32)
        scale_t = torch.as_tensor(scale, dtype=torch.float32)
        low_t = torch.as_tensor(lower_bounds, dtype=torch.float32)
        high_t = torch.as_tensor(upper_bounds, dtype=torch.float32)

        # 1. Standardize the bounds into latent Z-score space
        norm_lower = (low_t - mean_t) / scale_t
        norm_upper = (high_t - mean_t) / scale_t

        if pad > 0:
            mean_t = torch.cat([torch.zeros(pad, dtype=torch.float32), mean_t])
            scale_t = torch.cat([torch.ones(pad, dtype=torch.float32), scale_t])
            norm_lower = torch.cat(
                [torch.full((pad,), -torch.inf, dtype=torch.float32), norm_lower]
            )
            norm_upper = torch.cat(
                [torch.full((pad,), torch.inf, dtype=torch.float32), norm_upper]
            )

        self.register_buffer("mean", mean_t)
        self.register_buffer("scale", scale_t)
        self.register_buffer("norm_lower", norm_lower)
        self.register_buffer("norm_upper", norm_upper)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        if self.training:
            # 1. TRAINING: Clamp in normalized space; output stays normalized for loss computation
            return smooth_clamp(Z, self.norm_lower, self.norm_upper)
        else:
            # 2. INFERENCE: Hard clamp in normalized space, then unscale to physical units
            Z_clamped = torch.clamp(Z, self.norm_lower, self.norm_upper)
            return Z_clamped * self.scale + self.mean


def get_thermo_model(transform: BoundedOutputTransform):
    if not Path("chemeleon_mp.pt").exists():
        from urllib.request import urlretrieve

        urlretrieve(
            r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
            "chemeleon_mp.pt",
        )
    chemeleon_mp = torch.load("chemeleon_mp.pt", weights_only=True)
    mp = chemprop_nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
    mp.load_state_dict(chemeleon_mp["state_dict"])
    return models.MPNN(
        mp,
        chemprop_nn.MeanAggregation(),
        chemprop_nn.RegressionFFN(
            output_transform=transform,
            input_dim=mp.output_dim,
            activation=nn.GELU(),
            n_layers=2,
            hidden_dim=512,
            n_tasks=len(transform.mean),
        ),
        False,
        [chemprop_nn.metrics.RMSE(), chemprop_nn.metrics.MAE()],
    )


def get_kinetics_model(transform: BoundedOutputTransform):
    mp = chemprop_nn.BondMessagePassing(
        d_v=RIGR_RXN_FEATURIZER.atom_fdim,
        d_e=RIGR_RXN_FEATURIZER.bond_fdim,
    )
    return models.MPNN(
        mp,
        chemprop_nn.MeanAggregation(),
        chemprop_nn.RegressionFFN(
            input_dim=mp.output_dim,
            output_transform=transform,
            activation=nn.GELU(),
            n_layers=2,
            hidden_dim=512,
            n_tasks=len(transform.mean),
            # kinetics data has some strange outliers, thus huber loss
            criterion=HuberMetric(),
        ),
        False,
        [HuberMetric(), chemprop_nn.metrics.RMSE(), chemprop_nn.metrics.MAE()],
    )
