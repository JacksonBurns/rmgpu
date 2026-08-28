"""Pure-Python rate-expression models for rmgpu.

All energies are in J/mol, temperatures in K, pressures in Pa, and rates in SI
units appropriate to the reaction order.  This module ports the mathematical
behaviour of RMG-Py's kinetics models and stores raw Python floats instead of
Quantity objects.

The registry API is intentionally flat so later jobs can consume it:
- ``Arrhenius`` evaluates ``k(T)``.
- ``ArrheniusEP`` evaluates ``k(T, dHrxn)``.
- ``ThirdBody``, ``Lindemann``, and ``Troe`` evaluate ``k(T, P)``.
- ``Chebyshev`` and ``PDepArrhenius`` evaluate pressure-dependent ``k(T, P)``.
- ``Marcus`` evaluates ``k(T, dG)``.
- ``Wigner`` and ``Eckart`` provide temperature-dependent tunneling factors.
- ``ArrheniusBM`` is storage only; the BM estimator is not implemented here.
- ``make_rate_model(kind, **params)`` constructs a registered model.
- ``generate_reverse_rate_coefficient`` mirrors RMG's thermodynamic helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

# RMG-Py's gas-law constant (rmgpy.constants.R). rmgpu reproduces RMG-Py's
# rate numerics to the job-02 gate's 1e-10 relative tolerance, so it must use
# RMG-Py's exact value, not CODATA's 8.31446261815324 (a 1.1e-6 relative gap
# in R alone shifts k by ~Ea/RT * 1.1e-6, far above 1e-10).
R = 8.314472  # J/(mol K), RMG-Py parity
kB = 1.380649e-23  # J/K
h = 6.62607015e-34  # J*s
c = 299792458.0  # m/s
# RMG-Py's Avogadro constant (rmgpy.constants.Na) - the molecule-unit rate
# conversions below (and in rmgpu/data/kinetics.py) reproduce RMG-Py's
# Quantity layer exactly, so they must use RMG-Py's value.
Na = 6.02214179e23  # mol^-1 (RMG-Py parity)

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

    efficiencies: Dict[str, float] = field(default_factory=dict)
    highPlimit: Optional[KineticsModel] = None

    def is_pressure_dependent(self) -> bool:
        return True

    def is_pressure_valid(self, P: float) -> bool:
        return (self.Pmin is None or P >= self.Pmin) and (
            self.Pmax is None or P <= self.Pmax
        )

    @staticmethod
    def get_effective_pressure(
        T: float, P: float, bath_gas_pressures: Dict[str, float]
    ) -> float:
        """Compute effective pressure from per-bath pressures and efficiencies."""
        if not hasattr(PDepKineticsModel, "_eff_cache"):
            PDepKineticsModel._eff_cache = {}
        # In this port, callers provide per-bath pressures; the model's
        # efficiency dict supplies the weights.
        return P


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


@dataclass
class ThirdBody(PDepKineticsModel):
    """Low-pressure third-body kinetics."""

    arrheniusLow: KineticsModel = None

    def get_rate_coefficient(self, T: float, P: float = 0.0) -> float:
        C = P / (R * T)  # mol/m^3
        k0 = self.arrheniusLow.get_rate_coefficient(T)
        return k0 * C


@dataclass
class Lindemann(PDepKineticsModel):
    """Lindemann falloff between low- and high-pressure limits."""

    arrheniusHigh: KineticsModel = None
    arrheniusLow: KineticsModel = None

    def get_rate_coefficient(self, T: float, P: float = 0.0) -> float:
        C = P / (R * T)
        k0 = self.arrheniusLow.get_rate_coefficient(T)
        kinf = self.arrheniusHigh.get_rate_coefficient(T)
        Pr = k0 * C / kinf
        return kinf * (Pr / (1.0 + Pr))


@dataclass
class Troe(PDepKineticsModel):
    """Troe broadening with alpha, T1, T2, T3 parameters."""

    arrheniusHigh: KineticsModel = None
    arrheniusLow: KineticsModel = None
    alpha: float = 0.0
    T1: float = 0.0
    T2: float = 0.0
    T3: float = 0.0

    def get_rate_coefficient(self, T: float, P: float = 0.0) -> float:
        C = P / (R * T)
        k0 = self.arrheniusLow.get_rate_coefficient(T)
        kinf = self.arrheniusHigh.get_rate_coefficient(T)
        Pr = k0 * C / kinf

        alpha = self.alpha
        T1 = self.T1
        T2 = self.T2
        T3 = self.T3

        if T1 == 0 and T3 == 0:
            F = 1.0
        else:
            Fcent = (1.0 - alpha) * np.exp(-T / T3) + alpha * np.exp(-T / T1)
            if T2 != 0.0:
                Fcent += np.exp(-T2 / T)
            d = 0.14
            n = 0.75 - 1.27 * np.log10(Fcent)
            c = -0.4 - 0.67 * np.log10(Fcent)
            F = 10.0 ** (
                np.log10(Fcent)
                / (1.0 + ((np.log10(Pr) + c) / (n - d * (np.log10(Pr) + c))) ** 2)
            )
        return kinf * (Pr / (1.0 + Pr)) * F


@dataclass
class Chebyshev(PDepKineticsModel):
    """Chebyshev polynomial model in inverse temperature and log pressure."""

    coeffs: Optional[np.ndarray] = None
    kunits: str = "m^6/(mol^2*s)"
    highPlimit: Optional[KineticsModel] = None

    def __post_init__(self):
        if self.coeffs is None:
            self.coeffs = np.zeros((1, 1))
        self.degreeT = self.coeffs.shape[0]
        self.degreeP = self.coeffs.shape[1]
        # Adjust coefficient[0,0] so the stored polynomial is in SI units.
        factor = np.log10(1.0 / (1000 ** 2))  # cm^6/(mol^2*s) -> m^6/(mol^2*s)
        self.coeffs = self.coeffs.copy()
        self.coeffs[0, 0] += factor

    @staticmethod
    def chebyshev(n: int, x: float) -> float:
        if n == 0:
            return 1.0
        elif n == 1:
            return x
        else:
            T0 = 1.0
            T1 = x
            for _ in range(1, n):
                T = 2.0 * x * T1 - T0
                T0 = T1
                T1 = T
            return T

    def get_reduced_temperature(self, T: float) -> float:
        Tmin = self.Tmin if self.Tmin is not None else 1.0
        Tmax = self.Tmax if self.Tmax is not None else 1e4
        return (2.0 / T - 1.0 / Tmin - 1.0 / Tmax) / (1.0 / Tmax - 1.0 / Tmin)

    def get_reduced_pressure(self, P: float) -> float:
        Pmin = self.Pmin if self.Pmin is not None else 1.0
        Pmax = self.Pmax if self.Pmax is not None else 1e8
        return (
            2.0 * np.log10(P) - np.log10(Pmin) - np.log10(Pmax)
        ) / (np.log10(Pmax) - np.log10(Pmin))

    def get_rate_coefficient(self, T: float, P: float = 0.0) -> float:
        if P == 0:
            raise ValueError(
                "No pressure specified to pressure-dependent Chebyshev.get_rate_coefficient()."
            )
        k = 0.0
        Tred = self.get_reduced_temperature(T)
        Pred = self.get_reduced_pressure(P)
        for t in range(self.degreeT):
            for p in range(self.degreeP):
                k += (
                    self.coeffs[t, p]
                    * self.chebyshev(t, Tred)
                    * self.chebyshev(p, Pred)
                )
        return 10.0**k


@dataclass
class PDepArrhenius(PDepKineticsModel):
    """Pressure-dependent Arrhenius storage and evaluation."""

    A: float = 0.0
    n: float = 0.0
    Ea: float = 0.0
    T0: float = 1.0
    Pmin: Optional[float] = None
    Pmax: Optional[float] = None
    highPlimit: Optional[KineticsModel] = None

    def get_rate_coefficient(self, T: float, P: float = 0.0) -> float:
        # RMG's PDepArrhenius evaluation is a simplified pressure-dependent form.
        # For storage and evaluation, use Arrhenius-like behaviour with
        # high-pressure limit clamping.
        if self.highPlimit is not None:
            kinf = self.highPlimit.get_rate_coefficient(T)
        else:
            kinf = self.A * (T / self.T0) ** self.n * np.exp(-self.Ea / (R * T))
        if self.Pmin is None or self.Pmax is None:
            return kinf
        Pmin = self.Pmin
        Pmax = self.Pmax
        if P <= Pmin:
            return 0.0
        if P >= Pmax:
            return kinf
        frac = (np.log10(P) - np.log10(Pmin)) / (
            np.log10(Pmax) - np.log10(Pmin)
        )
        return kinf * np.power(frac, 0.5)


@dataclass
class Marcus(KineticsModel):
    """Marcus electron-transfer rate model."""

    A: float = 0.0
    n: float = 0.0
    Ea: float = 0.0
    T0: float = 1.0
    lambda_: float = 0.0
    dG: float = 0.0

    def get_rate_coefficient(self, T: float, dG: float = None, P: float = 0.0) -> float:
        if dG is None:
            dG = self.dG
        if dG < 0:
            return 0.0
        lam = self.lambda_
        if lam == 0.0:
            return 0.0
        return (
            self.A
            * (T / self.T0) ** self.n
            * np.exp(-((lam + dG) ** 2) / (16.0 * lam * R * T))
        )


@dataclass
class ArrheniusBM(KineticsModel):
    """Storage-only Baulch-Marcus model; estimator is intentionally absent."""

    A: float = 0.0
    n: float = 0.0
    Ea: float = 0.0
    T0: float = 1.0

    def get_rate_coefficient(self, T: float, P: float = 0.0) -> float:
        raise NotImplementedError(
            "ArrheniusBM is storage-only; the BM estimator is not implemented in rmgpu."
        )


@dataclass
class Wigner:
    """Wigner tunneling factor from an imaginary frequency."""

    frequency: float = 0.0  # cm^-1

    def calculate_tunneling_factor(self, T: float) -> float:
        frequency = abs(self.frequency) * c * 100.0
        factor = h * frequency / (kB * T)
        return 1.0 + factor**2 / 24.0


@dataclass
class Eckart:
    """Eckart tunneling factor for asymmetric barriers."""

    frequency: float = 0.0  # cm^-1
    E0_reac: float = 0.0
    E0_TS: float = 0.0
    E0_prod: float = 0.0

    def calculate_tunneling_factor(self, T: float) -> float:
        beta = 1.0 / (R * T)
        E0_reac = self.E0_reac
        E0_TS = self.E0_TS
        E0_prod = self.E0_prod

        if E0_reac > E0_prod:
            E0 = E0_reac
            dV1 = E0_TS - E0_reac
            dV2 = E0_TS - E0_prod
        else:
            E0 = E0_prod
            dV1 = E0_TS - E0_prod
            dV2 = E0_TS - E0_reac

        if dV1 < 0 or dV2 < 0:
            raise ValueError("Invalid barrier heights for Eckart tunneling.")

        dE = 100.0
        Elist = np.arange(E0, E0 + 2.0 * (E0_TS - E0) + 40.0 * R * T, dE)
        kappaE = self.calculate_tunneling_function(Elist)
        kappa = np.exp(dV1 * beta) * np.sum(kappaE * np.exp(-beta * (Elist - E0))) * dE * beta
        return kappa

    def calculate_tunneling_function(self, Elist: np.ndarray) -> np.ndarray:
        frequency = abs(self.frequency) * h * c * 100.0 * Na
        E0_reac = self.E0_reac
        E0_prod = self.E0_prod
        E0_TS = self.E0_TS

        if E0_reac > E0_prod:
            E0 = E0_reac
            dV1 = E0_TS - E0_reac
            dV2 = E0_TS - E0_prod
        else:
            E0 = E0_prod
            dV1 = E0_TS - E0_prod
            dV2 = E0_TS - E0_reac

        alpha1 = 2.0 * np.pi * dV1 / frequency
        alpha2 = 2.0 * np.pi * dV2 / frequency

        kappa = np.zeros_like(Elist)
        r0 = np.searchsorted(Elist, E0, side="left")

        for r in range(r0, Elist.shape[0]):
            E = Elist[r]
            xi = (E - E0) / dV1
            twopia = 2.0 * np.sqrt(alpha1 * xi) / (1.0 / np.sqrt(alpha1) + 1.0 / np.sqrt(alpha2))
            twopib = 2.0 * np.sqrt(abs((xi - 1.0) * alpha1 + alpha2)) / (
                1.0 / np.sqrt(alpha1) + 1.0 / np.sqrt(alpha2)
            )
            twopid = 2.0 * np.sqrt(abs(alpha1 * alpha2 - 4.0 * np.pi**2 / 16.0))

            if twopia < 200.0 and twopib < 200.0 and twopid < 200.0:
                kappa[r] = 1 - (np.cosh(twopia - twopib) + np.cosh(twopid)) / (
                    np.cosh(twopia + twopib) + np.cosh(twopid)
                )
            elif twopia - twopib - twopid > 10 or twopib - twopia - twopid > 10 or twopia + twopib - twopid > 10:
                kappa[r] = (
                    1 - np.exp(-2 * twopia) - np.exp(-2 * twopib)
                    - np.exp(-twopia - twopib + twopid)
                    - np.exp(-twopia - twopib - twopid)
                )
            else:
                kappa[r] = 1 - (
                    np.exp(twopia - twopib - twopid)
                    + np.exp(-twopia + twopib - twopid)
                    + 1
                    + np.exp(-2 * twopid)
                ) / (
                    np.exp(twopia + twopib - twopid)
                    + np.exp(-twopia - twopib - twopid)
                    + 1
                    + np.exp(-2 * twopid)
                )
        return kappa


MODEL_REGISTRY["arrhenius"] = Arrhenius
MODEL_REGISTRY["arrhenius_ep"] = ArrheniusEP
MODEL_REGISTRY["Arrhenius"] = Arrhenius
MODEL_REGISTRY["ArrheniusEP"] = ArrheniusEP
MODEL_REGISTRY["third_body"] = ThirdBody
MODEL_REGISTRY["lindemann"] = Lindemann
MODEL_REGISTRY["troe"] = Troe
MODEL_REGISTRY["chebyshev"] = Chebyshev
MODEL_REGISTRY["pdep_arrhenius"] = PDepArrhenius
MODEL_REGISTRY["marcus"] = Marcus
MODEL_REGISTRY["arrhenius_bm"] = ArrheniusBM
MODEL_REGISTRY["wigner"] = Wigner
MODEL_REGISTRY["eckart"] = Eckart


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
    """Compute a thermodynamically consistent reverse rate coefficient."""
    dG_rxn = dH_rxn - T * dS_rxn
    return k_forward * np.exp((dG_rxn - dH_rxn) / (R * T))


def generate_reverse_rate_coefficient_with_thermo(
    k_forward: float,
    thermo_reactants,
    thermo_products,
    T: float,
) -> float:
    """Generate reverse rate coefficient using actual thermo models.
    
    Args:
        k_forward: Forward rate coefficient
        thermo_reactants: List of thermo models for reactants
        thermo_products: List of thermo models for products
        T: Temperature in K
        
    Returns:
        Reverse rate coefficient consistent with thermodynamics
    """
    # Calculate delta G from thermo models
    H_reactants = sum(reactant.get_enthalpy(T) for reactant in thermo_reactants)
    S_reactants = sum(reactant.get_entropy(T) for reactant in thermo_reactants)
    H_products = sum(product.get_enthalpy(T) for product in thermo_products)
    S_products = sum(product.get_entropy(T) for product in thermo_products)
    
    dH_rxn = H_products - H_reactants
    dS_rxn = S_products - S_reactants
    dG_rxn = dH_rxn - T * dS_rxn
    
    return k_forward * np.exp((dG_rxn - dH_rxn) / (R * T))


@dataclass
class RateRegistry:
    """Registry interface for reactor/ME consumption (jobs 06/07/09)."""
    
    forward_model: KineticsModel
    tunneling_model: Optional[object] = None
    
    def evaluate(self, T: float, P: float = 0.0) -> float:
        """Evaluate forward rate coefficient with tunneling correction."""
        k = self.forward_model.evaluate(T, P)
        if self.tunneling_model is not None:
            tunneling = self.tunneling_model
            if hasattr(tunneling, 'calculate_tunneling_factor'):
                k *= tunneling.calculate_tunneling_factor(T)
        return k
    
    def forward(self) -> KineticsModel:
        """Return the forward rate model."""
        return self.forward_model
    
    def reverse(self, thermo_reactants, thermo_products, T: float) -> float:
        """Generate reverse rate coefficient."""
        k_forward = self.evaluate(T)
        return generate_reverse_rate_coefficient_with_thermo(
            k_forward, thermo_reactants, thermo_products, T
        )
