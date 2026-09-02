"""Reactor definitions ported from RMG-Py solver.

Termination criteria mirror RMG-Py's termination.py.
Reactor dataclasses mirror simple.pyx (SimpleReactor, ConstantVReactor,
ConstantTPReactor) and base.pyx maths (mole balance dN/dt = nu * r).

SI units internally: J, K, Pa, mol.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List

from rmgpu.units import Quantity


@dataclass
class TerminationTime:
    """Terminate simulation at a given time."""
    time: Quantity

    def __post_init__(self):
        if not isinstance(self.time, Quantity):
            self.time = Quantity(self.time, "s")


@dataclass
class TerminationConversion:
    """Terminate when fractional conversion of a species is reached."""
    species: str
    conversion: float


@dataclass
class TerminationRateRatio:
    """Terminate when characteristic rate falls to a fraction of its maximum."""
    ratio: float


@dataclass
class SimpleReactor:
    """Isothermal batch reactor at constant T and P.

    Mirrors RMG-Py rmgpy/solver/simple.pyx SimpleReactor.
    """
    temperature: Quantity
    pressure: Quantity
    initial_mole_fractions: Dict[str, float]
    termination: Optional[List] = None
    n_sims: int = 1
    sensitivity: Optional[List[str]] = None
    sensitivity_threshold: Optional[float] = None
    constant_species: Optional[List[str]] = None
    balance_species: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.temperature, Quantity):
            self.temperature = Quantity(self.temperature, "K")
        if not isinstance(self.pressure, Quantity):
            self.pressure = Quantity(self.pressure, "Pa")


@dataclass
class ConstantVReactor:
    """Constant volume ideal gas reactor.

    Mirrors RMG-Py rmgpy/solver/simple.pyx ConstantVReactor.
    """
    temperature: Quantity
    pressure: Quantity
    initial_mole_fractions: Dict[str, float]
    termination: Optional[List] = None
    balance_species: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.temperature, Quantity):
            self.temperature = Quantity(self.temperature, "K")
        if not isinstance(self.pressure, Quantity):
            self.pressure = Quantity(self.pressure, "Pa")


@dataclass
class ConstantTPReactor:
    """Constant T/P reactor with energy balance.

    Mirrors RMG-Py rmgpy/solver/simple.pyx ConstantTPReactor.
    The energy balance ODE is ported from simple.pyx/base.pyx.
    """
    temperature: Quantity
    pressure: Quantity
    initial_mole_fractions: Dict[str, float]
    termination: Optional[List] = None
    balance_species: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.temperature, Quantity):
            self.temperature = Quantity(self.temperature, "K")
        if not isinstance(self.pressure, Quantity):
            self.pressure = Quantity(self.pressure, "Pa")


# Stub types for later jobs
class LiquidReactor:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("LiquidReactor is implemented in job 11")

class MBSampledReactor:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("MBSampledReactor is implemented in job 11")

class SurfaceReactor:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SurfaceReactor is implemented in job 12")
