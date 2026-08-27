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


class BoundedOutputTransform(nn.Module):
    def __init__(
        self,
        mean: ArrayLike,
        scale: ArrayLike,
        pad: int = 0,
        lower_bounds: ArrayLike | None = None,
        upper_bounds: ArrayLike | None = None,
    ):
        super().__init__()

        mean = torch.cat([torch.zeros(pad), torch.tensor(mean, dtype=torch.float)])
        scale = torch.cat([torch.ones(pad), torch.tensor(scale, dtype=torch.float)])

        self.register_buffer(
            "lower_bounds", torch.tensor(lower_bounds, dtype=torch.float)
        )
        self.register_buffer(
            "upper_bounds", torch.tensor(upper_bounds, dtype=torch.float)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.training:
            # During training, apply boundedness constraints in a differentiable way
            return smooth_clamp(X, self.lower_bounds, self.upper_bounds)
        else:
            # During inference, apply boundedness constraints in a non-differentiable way (clamping)
            return torch.clamp(X, self.lower_bounds, self.upper_bounds)


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
            n_tasks=len(transform.lower_bounds),
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
            n_tasks=len(transform.lower_bounds),
            # kinetics data has some strange outliers, thus huber loss
            criterion=HuberMetric(),
        ),
        False,
        [HuberMetric(), chemprop_nn.metrics.RMSE(), chemprop_nn.metrics.MAE()],
    )
