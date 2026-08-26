"""Pure-Python rate-expression models for rmgpu.

All energies are in J/mol, temperatures in K, rates in SI units appropriate to
the reaction order.  This module ports the mathematical behaviour of RMG-Py's
``Arrhenius`` and ``ArrheniusEP`` classes, but stores raw Python floats instead of
Quantity objects.

The registry API is intentionally flat so later jobs can consume it:
- ``Arrhenius`` evaluates ``k(T)``.
- ``ArrheniusEP`` evaluates ``k(T, dHrxn)``.
- ``make_rate_model(kind, **params)`` constructs the correct class from a string.
- ``generate_reverse_rate_coefficient`` mirrors RMG's thermodynamic-consistency
  helper using a duck-typed thermo interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

R = 8.31446261815324  # J/(mol K), CODATA value used by RMG-Py.

# Registry of model classes.  The keys are the identifiers that callers are
# expected to pass to ``make_rate_model``.
MODEL_REGISTRY = {}


@dataclass
class KineticsModel:
    """Common attributes and helpers for pressure-independent kinetics models."""

    Tmin: Optional[float] = None
    Tmax: Optional[float] = None
    Pmin: Optional[float] = None
    Pmax: Optional[float] = None
    uncertainty: Optional[float] = None
    comment: str = ""

    def is_pressure_dependent(self) -> bool:
        return False

    def is_temperature_valid(self, T: float) -> bool:
        return (self.Tmin is None or T >= self.Tmin) and (
            self.Tmax is None or T <= self.Tmax
        )

    def get_rate_coefficient(self, T: float, P: float = 0.0, **kwargs) -> float:
        raise NotImplementedError(
            f"KineticsModel {self.__class__.__name__} does not implement get_rate_coefficient"
        )

    def evaluate(self, T: float, P: float = 0.0, **kwargs) -> float:
        return self.get_rate_coefficient(T, P, **kwargs)


@dataclass
class PDepKineticsModel(KineticsModel):
    """Common attributes for pressure-dependent kinetics models."""

    def is_pressure_dependent(self) -> bool:
        return True

    def is_pressure_valid(self, P: float) -> bool:
        return (self.Pmin is None or P >= self.Pmin) and (
            self.Pmax is None or P <= self.Pmax
        )


@dataclass
class Arrhenius(KineticsModel):
    """Modified Arrhenius model: k(T) = A * (T/T0)^n * exp(-Ea/(R*T))."""

    A: float = 0.0
    n: float = 0.0
    Ea: float = 0.0
    T0: float = 1.0

    def get_rate_coefficient(self, T: float, P: float = 0.0) -> float:
        return self.A * (T / self.T0) ** self.n * np.exp(-self.Ea / (R * T))

    def change_t0(self, T0: float) -> None:
        """Change reference temperature and adjust A to preserve the rate."""
        self.A *= (T0 / self.T0) ** self.n
        self.T0 = T0


@dataclass
class ArrheniusEP(KineticsModel):
    """Arrhenius model with Evans-Polanyi activation energy."""

    A: float = 0.0
    n: float = 0.0
    alpha: float = 0.0
    E0: float = 0.0
    T0: float = 1.0

    def get_activation_energy(self, dHrxn: float) -> float:
        Ea = self.alpha * dHrxn + self.E0
        if self.E0 > 0.0:
            if dHrxn < 0.0 and Ea < 0.0:
                Ea = 0.0
            elif dHrxn > 0.0 and Ea < dHrxn:
                Ea = dHrxn
        return Ea

    def get_rate_coefficient(self, T: float, dHrxn: float = 0.0, P: float = 0.0) -> float:
        Ea = self.get_activation_energy(dHrxn)
        return self.A * (T / self.T0) ** self.n * np.exp(-Ea / (R * T))

    def to_arrhenius(self, dHrxn: float) -> Arrhenius:
        """Convert to a fixed Arrhenius model for a given reaction enthalpy."""
        return Arrhenius(
            A=self.A,
            n=self.n,
            Ea=self.get_activation_energy(dHrxn),
            T0=1.0,
            Tmin=self.Tmin,
            Tmax=self.Tmax,
            Pmin=self.Pmin,
            Pmax=self.Pmax,
            uncertainty=self.uncertainty,
            comment=self.comment,
        )


MODEL_REGISTRY["arrhenius"] = Arrhenius
MODEL_REGISTRY["arrhenius_ep"] = ArrheniusEP
MODEL_REGISTRY["Arrhenius"] = Arrhenius
MODEL_REGISTRY["ArrheniusEP"] = ArrheniusEP


def make_rate_model(kind: str, **params) -> KineticsModel:
    """Construct a rate model from a registry key and parameters."""
    try:
        cls = MODEL_REGISTRY[kind]
    except KeyError as exc:
        raise KeyError(f"Unknown rate model type: {kind!r}") from exc
    return cls(**params)


def generate_reverse_rate_coefficient(
    k_forward: float,
    dH_rxn: float,
    dS_rxn: float,
    T: float,
) -> float:
    """Compute a thermodynamically consistent reverse rate coefficient.

    This follows the same relation used in RMG-Py's model.pyx:
    k_reverse = k_forward * exp((dG_rxn / (R*T)) - (dH_rxn / (R*T)))
    where dG_rxn = dH_rxn - T*dS_rxn.
    """
    dG_rxn = dH_rxn - T * dS_rxn
    return k_forward * np.exp((dG_rxn - dH_rxn) / (R * T))
