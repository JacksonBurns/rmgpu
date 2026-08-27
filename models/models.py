# Inference-only model definitions for the rmgpu ML estimators.
#
# This module is copied from the ml-fitting repo and stripped of everything that
# only exists to BUILD or TRAIN a model (the get_thermo_model / get_kinetics_model
# constructors and the pretrained-base download). It keeps ONLY the definitions
# that a trained checkpoint references at load time and that predict.py needs:
#   - the two featurizers (CHEMELEON_MOL_FEATURIZER, RIGR_RXN_FEATURIZER)
#   - BoundedOutputTransform (registered in the checkpoint's output transform)
#   - HuberMetric (registered in the kinetics model's FFN/metrics graph)
#
# WHY this file must exist (and be importable as top-level module `models`):
# chemprop saves the model with torch.save, which pickles class references by
# module path. BoundedOutputTransform and HuberMetric are referenced as
# `models.BoundedOutputTransform` / `models.HuberMetric`. So loading a checkpoint
# requires a module importable as `models` that defines those exact classes with
# the same buffer layout (lower_bounds/upper_bounds). If you rename this file or
# move the classes, the checkpoints will not load.
#
# Model training / fitting is OUT OF SCOPE for rmgpu (PLAN.md 8a.3): this package
# consumes these checkpoints; it never re-trains them.

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
