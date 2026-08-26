"""Data-only entry classes for the rmgpu database layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ThermoEntry:
    """A thermodynamic entry with either Wilhoit or NASA7 coefficients.

    For Wilhoit:
        Tdata: list of 7 temperatures in K
        Cpdata: list of 7 heat capacities in cal/(mol*K)
        H298: enthalpy at 298 K in kcal/mol
        S298: entropy at 298 K in cal/(mol*K)
        Cp0, CpInf, a0, a1, a2, a3, B, H0, S0: Wilhoit coefficients

    For NASA7:
        nasa_polynomials: list of dicts, each containing:
            c1..c7: coefficients
            Tmin, Tmax: temperature bounds in K
        E0, Cp0, CpInf: NASA parameters
    """

    label: str
    short_description: str
    long_description: str
    Tdata_unit: str
    Cpdata_unit: str
    H298: float
    H298_unit: str
    S298: float
    S298_unit: str
    Tdata: list[float] = field(default_factory=list)
    Cpdata: list[float] = field(default_factory=list)
    # Wilhoit coefficients
    Cp0: Optional[float] = None
    CpInf: Optional[float] = None
    a0: Optional[float] = None
    a1: Optional[float] = None
    a2: Optional[float] = None
    a3: Optional[float] = None
    B: Optional[float] = None
    H0: Optional[float] = None
    S0: Optional[float] = None
    # NASA polynomial data
    nasa_polynomials: list[dict] = field(default_factory=list)
    nasa_Tmin: Optional[float] = None
    nasa_Tmax: Optional[float] = None
    nasa_T_unit: Optional[str] = None
    E0: Optional[float] = None
    E0_unit: Optional[str] = None

    @property
    def has_wilhoit(self) -> bool:
        """Return True if this entry has Wilhoit coefficients."""
        return self.Cp0 is not None and self.CpInf is not None

    @property
    def has_nasa(self) -> bool:
        """Return True if this entry has NASA polynomial coefficients."""
        return bool(self.nasa_polynomials)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for serialization."""
        result: dict[str, Any] = {
            "label": self.label,
            "short_description": self.short_description,
            "long_description": self.long_description,
            "Tdata_unit": self.Tdata_unit,
            "Cpdata_unit": self.Cpdata_unit,
            "H298": self.H298,
            "H298_unit": self.H298_unit,
            "S298": self.S298,
            "S298_unit": self.S298_unit,
            "Tdata": self.Tdata,
            "Cpdata": self.Cpdata,
            "model": "wilhoit" if self.has_wilhoit else "nasa" if self.has_nasa else "unknown",
        }
        if self.has_wilhoit:
            result["Cp0"] = self.Cp0
            result["CpInf"] = self.CpInf
            result["a0"] = self.a0
            result["a1"] = self.a1
            result["a2"] = self.a2
            result["a3"] = self.a3
            result["B"] = self.B
            result["H0"] = self.H0
            result["S0"] = self.S0
        if self.has_nasa:
            result["nasa_polynomials"] = self.nasa_polynomials
            result["nasa_Tmin"] = self.nasa_Tmin
            result["nasa_Tmax"] = self.nasa_Tmax
            result["nasa_T_unit"] = self.nasa_T_unit
            result["E0"] = self.E0
            result["E0_unit"] = self.E0_unit
        return result


@dataclass
class KineticsEntry:
    """A kinetics entry with one rate model and reference."""

    label: str
    short_description: str
    long_description: str
    reaction: dict[str, Any]
    rate_model: dict[str, Any]
    reference: Optional[str] = None
    reference_type: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for serialization."""
        return {
            "label": self.label,
            "short_description": self.short_description,
            "long_description": self.long_description,
            "reaction": self.reaction,
            "rate_model": self.rate_model,
            "reference": self.reference,
            "reference_type": self.reference_type,
        }
