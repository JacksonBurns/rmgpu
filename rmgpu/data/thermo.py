"""Thermo model implementations: Wilhoit and NASA7 polynomials.

All values are in SI units: J, mol, K.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Sequence

# Gas constant in J/(mol*K)
R = 8.314462618


class Wilhoit:
    """Wilhoit heat capacity model.

    Attributes:
        Cp0: Heat capacity at T=0 in J/(mol*K)
        CpInf: Heat capacity at T=infinity in J/(mol*K)
        a0, a1, a2, a3: Polynomial coefficients
        H0: Integration constant for enthalpy in J/mol
        S0: Integration constant for entropy in J/(mol*K)
        B: Characteristic temperature in K
        Tmin: Minimum valid temperature in K
        Tmax: Maximum valid temperature in K
    """

    def __init__(
        self,
        Cp0: float,
        CpInf: float,
        a0: float = 0.0,
        a1: float = 0.0,
        a2: float = 0.0,
        a3: float = 0.0,
        H0: float = 0.0,
        S0: float = 0.0,
        B: float = 500.0,
        Tmin: Optional[float] = None,
        Tmax: Optional[float] = None,
    ):
        self.Cp0 = Cp0
        self.CpInf = CpInf
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.H0 = H0
        self.S0 = S0
        self.B = B
        self.Tmin = Tmin
        self.Tmax = Tmax

    def get_heat_capacity(self, T: float) -> float:
        """Return Cp(T) in J/(mol*K)."""
        y = T / (T + self.B)
        return self.Cp0 + (self.CpInf - self.Cp0) * y * y * (
            1 + (y - 1) * (self.a0 + y * (self.a1 + y * (self.a2 + y * self.a3)))
        )

    def get_enthalpy(self, T: float) -> float:
        """Return H(T) in J/mol."""
        y = T / (T + self.B)
        return self.H0 + self.Cp0 * T - (self.CpInf - self.Cp0) * T * (
            y * y
            * ((3 * self.a0 + self.a1 + self.a2 + self.a3) / 6.0
               + (4 * self.a1 + self.a2 + self.a3) * y / 12.0
               + (5 * self.a2 + self.a3) * y * y / 20.0
               + self.a3 * y * y * y / 5.0)
            + (2 + self.a0 + self.a1 + self.a2 + self.a3)
            * (y / 2.0 - 1 + (1.0 / y - 1.0) * np.log(self.B + T))
        )

    def get_entropy(self, T: float) -> float:
        """Return S(T) in J/(mol*K)."""
        y = T / (T + self.B)
        return self.S0 + self.CpInf * np.log(T) - (self.CpInf - self.Cp0) * (
            np.log(y) + y * (1 + y * (self.a0 / 2.0 + y * (self.a1 / 3.0 + y * (self.a2 / 4.0 + y * self.a3 / 5.0))))
        )

    def get_free_energy(self, T: float) -> float:
        """Return G(T) = H(T) - T*S(T) in J/mol."""
        return self.get_enthalpy(T) - T * self.get_entropy(T)

    def __repr__(self) -> str:
        return (
            f"Wilhoit(Cp0={self.Cp0}, CpInf={self.CpInf}, "
            f"a0={self.a0}, a1={self.a1}, a2={self.a2}, a3={self.a3}, "
            f"H0={self.H0}, S0={self.S0}, B={self.B}, "
            f"Tmin={self.Tmin}, Tmax={self.Tmax})"
        )


class NASAPolynomial:
    """Single NASA 7-coefficient polynomial.

    Coefficients are in the standard NASA 7-coeff format:
        c0, c1, c2, c3, c4, c5, c6
    Optionally with cm2, cm1 for 9-coefficients.
    """

    def __init__(
        self,
        coeffs: Sequence[float],
        Tmin: float,
        Tmax: float,
    ):
        """
        Args:
            coeffs: List of 7 or 9 NASA coefficients.
            Tmin: Minimum temperature in K.
            Tmax: Maximum temperature in K.
        """
        if len(coeffs) == 7:
            self.c0, self.c1, self.c2, self.c3, self.c4, self.c5, self.c6 = coeffs
            self.cm2 = 0.0
            self.cm1 = 0.0
        elif len(coeffs) == 9:
            self.cm2, self.cm1, self.c0, self.c1, self.c2, self.c3, self.c4, self.c5, self.c6 = coeffs
        else:
            raise ValueError(f"Invalid number of NASA coefficients; expected 7 or 9, got {len(coeffs)}.")
        self.Tmin = Tmin
        self.Tmax = Tmax

    def is_temperature_valid(self, T: float) -> bool:
        """Return True if T is within [Tmin, Tmax]."""
        return self.Tmin <= T <= self.Tmax

    def get_heat_capacity(self, T: float) -> float:
        """Return Cp(T) in J/(mol*K)."""
        return (
            ((self.cm2 / T + self.cm1) / T + self.c0 + T * (self.c1 + T * (self.c2 + T * (self.c3 + self.c4 * T))))
            * R
        )

    def get_enthalpy(self, T: float) -> float:
        """Return H(T) in J/mol."""
        T2 = T * T
        T4 = T2 * T2
        return (
            ((-self.cm2 / T + self.cm1 * np.log(T)) / T + self.c0 + self.c1 * T / 2.0
             + self.c2 * T2 / 3.0 + self.c3 * T2 * T / 4.0 + self.c4 * T4 / 5.0 + self.c5 / T)
            * R * T
        )

    def get_entropy(self, T: float) -> float:
        """Return S(T) in J/(mol*K)."""
        T2 = T * T
        T4 = T2 * T2
        return (
            ((-self.cm2 / T / 2.0 - self.cm1) / T + self.c0 * np.log(T) + self.c1 * T
             + self.c2 * T2 / 2.0 + self.c3 * T2 * T / 3.0 + self.c4 * T4 / 4.0 + self.c6)
            * R
        )

    def __repr__(self) -> str:
        return (
            f"NASAPolynomial(coeffs=[{self.c0}, {self.c1}, {self.c2}, {self.c3}, "
            f"{self.c4}, {self.c5}, {self.c6}], Tmin={self.Tmin}, Tmax={self.Tmax})"
        )


class NASA:
    """NASA 7-coefficient heat capacity model with one or more polynomials."""

    def __init__(
        self,
        polynomials: Sequence[dict],
        Tmin: Optional[float] = None,
        Tmax: Optional[float] = None,
        E0: Optional[float] = None,
    ):
        """
        Args:
            polynomials: List of dicts, each with keys 'c1'..c7 (or 'cm2','cm1',c0..c6), 'Tmin', 'Tmax'.
            Tmin: Overall minimum temperature.
            Tmax: Overall maximum temperature.
            E0: Energy at zero Kelvin.
        """
        self.polynomials = [NASAPolynomial(p['coeffs'], p['Tmin'], p['Tmax']) for p in polynomials]
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.E0 = E0

    def select_polynomial(self, T: float) -> NASAPolynomial:
        """Select the appropriate polynomial for temperature T."""
        for poly in self.polynomials:
            if poly.is_temperature_valid(T):
                return poly
        raise ValueError(f"No valid NASA polynomial at temperature {T:g} K.")

    def get_heat_capacity(self, T: float) -> float:
        """Return Cp(T) in J/(mol*K)."""
        return self.select_polynomial(T).get_heat_capacity(T)

    def get_enthalpy(self, T: float) -> float:
        """Return H(T) in J/mol."""
        return self.select_polynomial(T).get_enthalpy(T)

    def get_entropy(self, T: float) -> float:
        """Return S(T) in J/(mol*K)."""
        return self.select_polynomial(T).get_entropy(T)

    def get_free_energy(self, T: float) -> float:
        """Return G(T) = H(T) - T*S(T) in J/mol."""
        return self.get_enthalpy(T) - T * self.get_entropy(T)

    def __repr__(self) -> str:
        return f"NASA(polynomials={self.polynomials}, Tmin={self.Tmin}, Tmax={self.Tmax})"
