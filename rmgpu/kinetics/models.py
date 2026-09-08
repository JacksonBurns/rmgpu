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
from typing import Any, Dict, List, Optional

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

# Pre-exponential volume-unit CGS -> SI conversion factors (m^3 per cm^3).
# RMG-Py's Quantity layer (rmgpy/quantity.py RateCoefficient.value_si) applies
# these when converting a stored rate to SI. The Chebyshev coefficient
# zeroth-order term c00 carries a log10(RateCoefficient(1.0, kunits).value_si)
# shift on file load (see Chebyshev.__post_init__); the driver's fit stores
# coefficients on the SI scale (RMG's fit_to_data converts K -> SI before
# fitting log10), so the fit path passes SI kunits and gets factor 1.
_KUNITS_CGS_TO_SI3 = {
    "cm^3/(mol*s)": 1.0e-6,
    "cm^6/(mol^2*s)": 1.0e-12,
    "cm^9/(mol^3*s)": 1.0e-18,
}


def kunits_to_si(kunits: str) -> float:
    """CGS -> SI pre-exponential volume factor for `kunits` (RMG quantity.py).

    SI or empty units -> 1.0 (no shift); the CGS volume units -> the m^3/cm^3
    factor to the reaction order's power. Anything else is treated as SI.
    """
    if kunits is None:
        return 1.0
    return _KUNITS_CGS_TO_SI3.get(str(kunits), 1.0)


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
    """Chebyshev polynomial model in inverse temperature and log pressure.

    The stored `coeffs` matrix holds `log10(k)` on the SI scale (RMG-Py's
    ``fit_to_data`` converts K -> SI before fitting ``log10``; the zeroth-order
    coefficient ``c00`` carries a ``log10(RateCoefficient(1.0, kunits).value_si)``
    shift on FILE LOAD so the stored polynomial is on the SI scale. See
    :meth:`__post_init__` and :meth:`get_rate_coefficient` (both SI).
    """

    coeffs: Optional[np.ndarray] = None
    kunits: str = ""          # the kunits the coeffs were fit/stored in (RMG)
    highPlimit: Optional[KineticsModel] = None

    def __post_init__(self):
        if self.coeffs is None:
            self.coeffs = np.zeros((1, 1))
        self.degreeT = self.coeffs.shape[0]
        self.degreeP = self.coeffs.shape[1]
        # RMG-Py (rmgpy/kinetics/chebyshev.pyx __init__): on load, the stored
        # polynomial is shifted so that c00 encodes the SI scale for the
        # model's `kunits`. The CGS -> SI factor is 1e-6 (cm^3), 1e-12 (cm^6),
        # or 1.0 for SI units. A coefficient fitted on the SI scale (the driver
        # path, see rmgpu/pdep/driver.py) stores SI kunits and gets factor 1.0,
        # so the fit is unaffected; only file-loaded CGS coefficients are shifted.
        # (The pre-step-06 code hard-coded -6 for ALL models regardless of
        # kunits, which silently corrupted any SI-fitted Chebyshev.)
        factor = kunits_to_si(self.kunits)
        if factor != 1.0:
            self.coeffs = self.coeffs.copy()
            self.coeffs[0, 0] += float(np.log10(factor))

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

    def fit_to_data(self, Tlist, Plist, K, kunits: str = "",
                    degreeT: int = 6, degreeP: int = 4,
                    Tmin: float = 0.0, Tmax: float = 0.0,
                    Pmin: float = 0.0, Pmax: float = 0.0) -> "Chebyshev":
        """RMG Chebyshev.fit_to_data (chebyshev.pyx:177): fit ``log10(K)`` on
        the reduced Chebyshev basis by least squares. K is SI (the driver
        passes SI kunits); the stored coeffs are therefore on the SI scale and
        ``get_rate_coefficient`` returns SI directly. The kunits are accepted
        for RMG-signature parity (RMG converts a CGS K to SI internally;
        rmgpu keeps SI throughout)."""
        Tlist = np.asarray(Tlist, dtype=float)
        Plist = np.asarray(Plist, dtype=float)
        K = np.asarray(K, dtype=float)
        nT, nP = len(Tlist), len(Plist)
        if nT <= degreeT or nP <= degreeP:
            raise ValueError(
                "The master equation data needs more temperature and pressure "
                "data points than the Chebyshev polynomial degree.")
        self.Tmin = float(Tmin) if Tmin else float(Tlist[0])
        self.Tmax = float(Tmax) if Tmax else float(Tlist[-1])
        self.Pmin = float(Pmin) if Pmin else float(Plist[0])
        self.Pmax = float(Pmax) if Pmax else float(Plist[-1])
        Tred = [self.get_reduced_temperature(t) for t in Tlist]
        Pred = [self.get_reduced_pressure(p) for p in Plist]
        A = np.zeros((nT * nP, degreeT * degreeP), float)
        b = np.zeros((nT * nP), float)
        for t1 in range(nT):
            for p1 in range(nP):
                for t2 in range(degreeT):
                    for p2 in range(degreeP):
                        A[p1 * nT + t1, p2 * degreeT + t2] = (
                            self.chebyshev(t2, Tred[t1])
                            * self.chebyshev(p2, Pred[p1]))
                b[p1 * nT + t1] = np.log10(K[t1, p1])
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=-1)
        coeffs = np.zeros((degreeT, degreeP), float)
        for t2 in range(degreeT):
            for p2 in range(degreeP):
                coeffs[t2, p2] = x[p2 * degreeT + t2]
        self.kunits = kunits
        self.coeffs = coeffs
        self.degreeT = degreeT
        self.degreeP = degreeP
        return self


def _fit_arrhenius(Tlist, klist, T0: float = 1.0) -> "Arrhenius":
    """RMG's 3-parameter Arrhenius least-squares fit (arrhenius.pyx fit_to_data):
    fit ``log(k) = log(A) + n*log(T/T0) - Ea/(R*T)`` by ``lstsq``. Returns an
    rmgpu ``Arrhenius`` with SI A (klist is SI) and Ea in J/mol. Mirrors
    rmgpy/kinetics/arrhenius.pyx:149 exactly (the coefficient of ``-1/(R*T)``
    is Ea in J/mol)."""
    Tlist = np.asarray(Tlist, dtype=float)
    klist = np.asarray(klist, dtype=float)
    if np.any(klist <= 0):
        raise ValueError("Arrhenius fit requires all-positive rate coefficients.")
    Am = np.zeros((len(Tlist), 3), float)
    Am[:, 0] = 1.0
    Am[:, 1] = np.log(Tlist / T0)
    Am[:, 2] = -1.0 / R / Tlist
    b = np.log(klist)
    x, _, _, _ = np.linalg.lstsq(Am, b, rcond=-1)
    return Arrhenius(A=float(np.exp(x[0])), n=float(x[1]),
                     Ea=float(x[2]), T0=float(T0))


@dataclass
class PDepArrhenius(PDepKineticsModel):
    """Pressure-dependent Arrhenius: an Arrhenius fit at EACH grid pressure,
    interpolated in log-log between adjacent pressures.

    Two storage shapes (RMG parity, rmgpy/kinetics/arrhenius.pyx PDepArrhenius):
    - RMG shape (the driver's fit): ``pressures`` (Pa, array) + ``arrhenius``
      (list of Arrhenius, one per pressure). ``fit_to_data`` populates these;
      ``get_rate_coefficient`` does RMG's log-log interpolation between the
      two adjacent pressures.
    - Legacy single-model shape (pre-step-06, kept for the job-02 assembly
      tests): ``A``/``n``/``Ea``/``T0``/``highPlimit`` + a Pmin->Pmax power
      ramp. Used only when ``arrhenius`` is empty.
    Both evaluate in SI (m^3/mol/s etc. for the reaction order).
    """

    # RMG shape (the driver's fit target)
    pressures: Optional[List[float]] = None
    arrhenius: Optional[List[Arrhenius]] = None
    # Legacy single-model shape (backward compat)
    A: float = 0.0
    n: float = 0.0
    Ea: float = 0.0
    T0: float = 1.0
    Pmin: Optional[float] = None
    Pmax: Optional[float] = None
    highPlimit: Optional[KineticsModel] = None

    def _adjacent(self, P: float):
        pressures = list(self.pressures)
        ilow, ihigh = 0, 0
        for i, p in enumerate(pressures):
            if p <= P:
                ilow = i
            if p >= P and ihigh == 0:
                ihigh = i
        if ihigh == 0:
            ihigh = len(pressures) - 1
        return pressures[ilow], pressures[ihigh], self.arrhenius[ilow], self.arrhenius[ihigh]

    def get_rate_coefficient(self, T: float, P: float = 0.0) -> float:
        if P == 0:
            raise ValueError(
                "No pressure specified to pressure-dependent PDepArrhenius.get_rate_coefficient().")
        # RMG shape: log-log interpolation between adjacent pressures.
        if self.arrhenius is not None and self.pressures is not None:
            Plow, Phigh, alow, ahigh = self._adjacent(P)
            if Plow == Phigh:
                return alow.get_rate_coefficient(T)
            klow = alow.get_rate_coefficient(T)
            khigh = ahigh.get_rate_coefficient(T)
            if klow == khigh == 0.0:
                return 0.0
            return klow * 10.0 ** (
                np.log10(P / Plow) / np.log10(Phigh / Plow)
                * np.log10(khigh / klow))
        # Legacy single-model shape (job-02 assembly tests).
        if self.highPlimit is not None:
            kinf = self.highPlimit.get_rate_coefficient(T)
        else:
            kinf = self.A * (T / self.T0) ** self.n * np.exp(-self.Ea / (R * T))
        if self.Pmin is None or self.Pmax is None:
            return kinf
        if P <= self.Pmin:
            return 0.0
        if P >= self.Pmax:
            return kinf
        frac = (np.log10(P) - np.log10(self.Pmin)) / (
            np.log10(self.Pmax) - np.log10(self.Pmin))
        return kinf * np.power(frac, 0.5)

    def fit_to_data(self, Tlist, Plist, K, kunits: str = "", T0: float = 298.0):
        """RMG PDepArrhenius.fit_to_data: fit an Arrhenius at EACH pressure
        column of K (K is SI, so A is stored SI), then log-log interpolate.
        Mirrors rmgpy/kinetics/arrhenius.pyx:906 (the kunits are accepted for
        RMG-signature parity but rmgpu stores SI throughout)."""
        Tlist = np.asarray(Tlist, dtype=float)
        Plist = np.asarray(Plist, dtype=float)
        K = np.asarray(K, dtype=float)
        pressures = list(Plist)
        arrhenius = []
        for i in range(len(Plist)):
            a = _fit_arrhenius(Tlist, K[:, i], T0)
            arrhenius.append(a)
        self.pressures = pressures
        self.arrhenius = arrhenius
        self.Pmin = float(Plist[0])
        self.Pmax = float(Plist[-1])
        self.Tmin = float(Tlist[0])
        self.Tmax = float(Tlist[-1])
        return self


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
