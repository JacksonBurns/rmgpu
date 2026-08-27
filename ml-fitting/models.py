from pathlib import Path

import torch
from torch import nn
from numpy.typing import ArrayLike

from chemprop import featurizers, models, nn as chemprop_nn
from chemprop.featurizers.atom import RIGRAtomFeaturizer
from chemprop.featurizers.bond import RIGRBondFeaturizer


CHEMELEON_MOL_FEATURIZER = featurizers.SimpleMoleculeMolGraphFeaturizer()
RIGR_RXN_FEATURIZER = featurizers.CondensedGraphOfReactionFeaturizer(
    atom_featurizer=RIGRAtomFeaturizer(), bond_featurizer=RIGRBondFeaturizer()
)


class BoundedUnscaleTransform(nn.Module):
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

        self.register_buffer("mean", mean.unsqueeze(0))
        self.register_buffer("scale", scale.unsqueeze(0))
        self.register_buffer(
            "lower_bounds", torch.tensor(lower_bounds, dtype=torch.float)
        )
        self.register_buffer(
            "upper_bounds", torch.tensor(upper_bounds, dtype=torch.float)
        )
        self.register_buffer(
            "scaled_lower_bounds",
            None
            if self.lower_bounds is None
            else (self.lower_bounds - self.mean) / self.scale,
        )
        self.register_buffer(
            "scaled_upper_bounds",
            None
            if self.upper_bounds is None
            else (self.upper_bounds - self.mean) / self.scale,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.training:
            # During training, apply boundedness constraints in a differentiable way
            return torch.clamp(X, self.scaled_lower_bounds, self.scaled_upper_bounds)
        else:
            # During inference, apply boundedness constraints and unscaling
            X_unscaled = X * self.scale + self.mean
            return torch.clamp(X_unscaled, self.lower_bounds, self.upper_bounds)


def get_thermo_model(transform: BoundedUnscaleTransform):
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
        chemprop_nn.RegressionFFN(output_transform=transform, input_dim=mp.output_dim),
        False,
        [chemprop_nn.metrics.RMSE(), chemprop_nn.metrics.MAE()],
    )


def get_kinetics_model(transform: BoundedUnscaleTransform):
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
        ),
        False,
        [chemprop_nn.metrics.RMSE(), chemprop_nn.metrics.MAE()],
    )
