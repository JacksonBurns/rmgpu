"""
rmgpu.statmech.modes
====================

Production port of the RMG-Py statistical-mechanics mode classes
(``rmgpy/statmech/{vibration,translation,rotation,torsion,schrodinger}.pyx``)
to pure Python/NumPy, for job-07/step-07 (the gate).

This is the REAL density-of-states machinery that the production pdep network
path uses (``rmgmode``). It replaces the step-01 sanity stub. The numerical
primitives (the Beyer-Swinehart convolution, the quantum harmonic-oscillator
DoS, the classical translation/rotation DoS) are ported line-for-line so the
rmgpu values reproduce RMG-Py's to machine precision (verified against the
non-circular oracle ``gates/baselines/propane_branching/statmech_dos_oracle.json``
and the 18 reactive species in ``statmech.json``; see tests/test_statmech_dos.py).

Constants match RMG-Py (rmgpy/constants) EXACTLY so the 1e-10/1e-6 parity gates
hold (see the note about R in the memory: rmgpy uses R=8.314472, NOT CODATA).

Energy convention: frequencies are stored in cm^-1, energies in J/mol,
temperatures in K, mass in kg (per molecule), moments of inertia in
kg*m^2 (SI). DoS is rho(E)*dE (the Beyer-Swinehart direct count) for the
quantum oscillator and the classical density times dE for translation/rotation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.special import factorial
from scipy.linalg import eig_banded

# --------------------------------------------------------------------------- #
# RMG-Py constants (rmgpy/constants) - EXACT values for parity.
# --------------------------------------------------------------------------- #
R = 8.314472                      # J/(mol K)   (rmgpy value, NOT CODATA)
kB = 1.3806504e-23                # J/K
h = 6.62606896e-34                # J s
hbar = 1.054571726e-34            # J s
c = 299792458.0                   # m/s
Na = 6.02214179e23                # 1/mol
amu = 1.660538921e-27             # kg
# h * c * 100 : converts a cm^-1 wavenumber (value_si) to J per photon *100...
# In RMG-Py: freq (cm^-1 number) * c (m/s) * 100 -> the per-oscillator energy
# spacing in J/mol is freq * h * c * 100 * Na.
_HC100 = h * c * 100.0            # = 1.9864455e-23 J*m  (per mol via *Na below)
_P0 = 101325.0                     # Pa (used by IdealGasTranslation)


# --------------------------------------------------------------------------- #
# Beyer-Swinehart convolution primitives (rmgpy/statmech/schrodinger.pyx).
# --------------------------------------------------------------------------- #
def convolve(rho1: np.ndarray, rho2: np.ndarray) -> np.ndarray:
    """Discrete convolution of two same-length DoS arrays (schrodinger.convolve)."""
    if rho1.shape[0] != rho2.shape[0]:
        raise ValueError("attempted to convolve mismatched lengths")
    nE = rho1.shape[0]
    rho = np.zeros_like(rho1)
    for i in range(nE):
        for j in range(i + 1):
            rho[i] += rho2[i - j] * rho1[j]
    return rho


def convolve_bs(e_list: np.ndarray, rho0: np.ndarray, energy: float,
                degeneracy: int = 1) -> np.ndarray:
    """
    Beyer-Swinehart direct count (schrodinger.convolve_bs). Convolve one
    evenly-spaced set of levels (spacing ``energy`` in J/mol, ``degeneracy``)
    into ``rho0`` on the energy grid ``e_list`` (J/mol above the ground state).
    """
    Emax = float(np.max(e_list))
    nE = e_list.shape[0]
    rho = rho0.copy()
    for i in range(nE):
        for j in range(i, nE):
            if e_list[j] - e_list[i] >= 0.9999 * energy:
                rho[j] += degeneracy * rho[i]
                break
    return rho


def convolve_bssr(e_list: np.ndarray, rho0: np.ndarray, energy,
                  degeneracy=None, n0: int = 0) -> np.ndarray:
    """
    Beyer-Swinehart-Stein-Rabinovitch direct count (schrodinger.convolve_bssr).
    Like ``convolve_bs`` but for UNEVENLY-spaced levels: ``energy(n)`` and
    ``degeneracy(n)`` are callables returning the J (J/mol) energy and the
    degeneracy of level ``n`` (starting at ``n0``). ``degeneracy`` defaults to 1.
    """
    if degeneracy is None:
        degeneracy = lambda n: 1
    Emax = float(np.max(e_list))
    nE = e_list.shape[0]
    rho = np.zeros_like(rho0)
    n = n0
    E_n = energy(n)
    g_n = degeneracy(n)
    while E_n < Emax:
        for i in range(nE):
            for j in range(i, nE):
                if e_list[j] - e_list[i] >= 0.9999 * E_n:
                    rho[j] += g_n * rho0[i]
                    break
        n += 1
        E_n = energy(n)
        g_n = degeneracy(n)
    return rho


# --------------------------------------------------------------------------- #
# Mode base class
# --------------------------------------------------------------------------- #
class Mode:
    def __init__(self, quantum: bool = True):
        self.quantum = quantum

    def get_heat_capacity(self, T: float) -> float:
        raise NotImplementedError

    def get_sum_of_states(self, e_list, sum_states_0=None) -> np.ndarray:
        raise NotImplementedError

    def get_density_of_states(self, e_list, dens_states_0=None) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# HarmonicOscillator  (rmgpy/statmech/vibration.pyx)
# --------------------------------------------------------------------------- #
class HarmonicOscillator(Mode):
    def __init__(self, frequencies, quantum: bool = True):
        super().__init__(quantum)
        self.frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))  # cm^-1

    # Per-oscillator energy in J/mol (RMG formula: freq * h * c * 100 * Na).
    def _energies(self):
        return self.frequencies * _HC100 * Na

    def get_partition_function(self, T: float) -> float:
        beta = 1.0 / (R * T)          # molar beta (RMG: 1/(kB*T) * Na == 1/(R*T))
        Q = 1.0
        for E in self._energies():
            if self.quantum:
                Q *= 1.0 / (1.0 - math.exp(-beta * E))
            else:
                Q *= 1.0 / (beta * E)
        return Q

    def get_heat_capacity(self, T: float) -> float:
        beta = 1.0 / (R * T)
        Cv = 0.0
        if self.quantum:
            for E in self._energies():
                x = beta * E
                if x > 500.0:
                    Cv += 0.0
                else:
                    exp_x = math.exp(x)
                    Cv += x * x * exp_x / (1 - exp_x) / (1 - exp_x)
        else:
            Cv = self.frequencies.shape[0]
        return Cv * R

    def get_sum_of_states(self, e_list, sum_states_0=None) -> np.ndarray:
        if self.quantum:
            sum_states = np.ones_like(e_list) if sum_states_0 is None else sum_states_0
            for E in self._energies():
                sum_states = convolve_bs(e_list, sum_states, E, 1)
        else:
            Nfreq = self.frequencies.shape[0]
            if sum_states_0 is not None:
                sum_states = convolve(sum_states_0, self.get_density_of_states(e_list))
            else:
                sum_states = e_list ** Nfreq / factorial(Nfreq)
                for E in self._energies():
                    sum_states = sum_states / E
        return sum_states

    def get_density_of_states(self, e_list, dens_states_0=None) -> np.ndarray:
        if self.quantum:
            dens_states = dens_states_0 if dens_states_0 is not None else np.zeros_like(e_list)
            if dens_states_0 is None:
                dens_states[0] = 1.0
            for E in self._energies():
                dens_states = convolve_bs(e_list, dens_states, E, 1)
        else:
            Nfreq = self.frequencies.shape[0]
            dE = e_list[1] - e_list[0]
            dens_states = e_list ** (Nfreq - 1) / factorial(Nfreq - 1) * dE
            for E in self._energies():
                dens_states = dens_states / E
            if dens_states_0 is not None:
                dens_states = convolve(dens_states_0, dens_states)
        return dens_states


# --------------------------------------------------------------------------- #
# IdealGasTranslation  (rmgpy/statmech/translation.pyx)
# --------------------------------------------------------------------------- #
class IdealGasTranslation(Mode):
    def __init__(self, mass: float, quantum: bool = False):
        super().__init__(quantum)
        self.mass = float(mass)  # kg per molecule

    def get_heat_capacity(self, T: float) -> float:
        if self.quantum:
            raise NotImplementedError
        return 2.5 * R

    def get_sum_of_states(self, e_list, sum_states_0=None) -> np.ndarray:
        if self.quantum:
            raise NotImplementedError
        e = e_list / Na
        qt = ((2 * math.pi * self.mass) / (h * h)) ** 1.5 / _P0
        if sum_states_0 is not None:
            return convolve(sum_states_0, self.get_density_of_states(e_list))
        return qt * e ** 2.5 / (math.sqrt(math.pi) * 15.0 / 8.0)

    def get_density_of_states(self, e_list, dens_states_0=None) -> np.ndarray:
        if self.quantum:
            raise NotImplementedError
        e = e_list / Na
        dE = e[1] - e[0]          # per-particle spacing (RMG divides e_list by Na first)
        qt = ((2 * math.pi * self.mass) / (h * h)) ** 1.5 / _P0
        dens = qt * e ** 1.5 / (math.sqrt(math.pi) * 0.75) * dE
        if dens_states_0 is not None:
            dens = convolve(dens_states_0, dens)
        return dens


# --------------------------------------------------------------------------- #
# Rotation  (rmgpy/statmech/rotation.pyx)
# --------------------------------------------------------------------------- #
class LinearRotor(Mode):
    def __init__(self, inertia: float = None, symmetry: int = 1, quantum: bool = True,
                 rotational_constant: float = None):
        super().__init__(quantum)
        self.symmetry = symmetry
        if inertia is None and rotational_constant is not None:
            # I from B (cm^-1): I = h / (8 pi^2 (B c 100))
            self.inertia = h / (8 * math.pi * math.pi * (rotational_constant * c * 100.0))
        else:
            self.inertia = float(inertia)  # kg m^2

    def _B_energy(self) -> float:
        # rotational constant in J/mol
        return hbar * hbar / (2 * self.inertia) * Na

    def get_heat_capacity(self, T: float) -> float:
        # classical limit (RMG default for a 1D linear rotor)
        return 1.0 * R

    def get_sum_of_states(self, e_list, sum_states_0=None) -> np.ndarray:
        B = self._B_energy()
        if sum_states_0 is not None:
            return convolve(sum_states_0, self.get_density_of_states(e_list))
        return e_list / B / self.symmetry

    def get_density_of_states(self, e_list, dens_states_0=None) -> np.ndarray:
        B = self._B_energy()
        dE = e_list[1] - e_list[0]
        dens = np.ones_like(e_list) * dE / B / self.symmetry
        if dens_states_0 is not None:
            dens = convolve(dens_states_0, dens)
        return dens


class NonlinearRotor(Mode):
    def __init__(self, inertia, symmetry: int = 1, quantum: bool = False):
        super().__init__(quantum)
        self.symmetry = symmetry
        self.inertia = np.atleast_1d(np.asarray(inertia, dtype=float))  # kg m^2

    def _B_energies(self) -> np.ndarray:
        return hbar * hbar / (2 * self.inertia) * Na

    def get_heat_capacity(self, T: float) -> float:
        if self.quantum:
            raise NotImplementedError
        return 0.5 * self.inertia.shape[0] * R

    def get_sum_of_states(self, e_list, sum_states_0=None) -> np.ndarray:
        if self.quantum:
            raise NotImplementedError
        if sum_states_0 is not None:
            return convolve(sum_states_0, self.get_density_of_states(e_list))
        B = self._B_energies()
        theta = float(np.prod(B))
        return 4.0 / 3.0 * e_list * np.sqrt(e_list / theta) / self.symmetry

    def get_density_of_states(self, e_list, dens_states_0=None) -> np.ndarray:
        if self.quantum:
            raise NotImplementedError
        dE = e_list[1] - e_list[0]
        B = self._B_energies()
        theta = float(np.prod(B))
        dens = 2.0 * np.sqrt(e_list / theta) / self.symmetry * dE
        if dens_states_0 is not None:
            dens = convolve(dens_states_0, dens)
        return dens


# --------------------------------------------------------------------------- #
# HinderedRotor  (rmgpy/statmech/torsion.pyx)
# --------------------------------------------------------------------------- #
class HinderedRotor(Mode):
    def __init__(self, inertia: float = None, symmetry: int = 1,
                 barrier: float = None, fourier=None, quantum: bool = True):
        super().__init__(quantum)
        self.symmetry = symmetry
        self.inertia = float(inertia)      # kg m^2
        self.barrier = float(barrier) if barrier is not None else None  # J/mol
        self.fourier = fourier            # (2, N) J/mol array or None
        self._energies = None             # J/mol eigenvalues, lazy

    def _B_energy(self) -> float:
        return hbar * hbar / (2 * self.inertia) * Na

    # ------------------------------------------------------------------ #
    # Quantum eigenproblem (rmgpy/statmech/torsion.pyx get_hamiltonian +
    # solve_schrodinger_equation). Banded lower-triangular Hamiltonian,
    # scipy.linalg.eig_banded.
    # ------------------------------------------------------------------ #
    def _hamiltonian(self, n_basis: int):
        M = n_basis // 2 if n_basis % 2 == 0 else (n_basis - 1) // 2
        if self.fourier is not None:
            coeffs = np.atleast_2d(np.asarray(self.fourier, dtype=float))
            V0 = -float(np.sum(coeffs[0, :]))
        else:
            coeffs = np.zeros((2, self.symmetry), float)
            V0 = 0.5 * self.barrier
            coeffs[0, self.symmetry - 1] = -V0
        H = np.zeros((coeffs.shape[1] + 1, 2 * M + 1), np.complex64)
        B = self._B_energy()
        col = 0
        for m in range(-M, M + 1):
            H[0, col] = B * m * m + V0
            for n in range(coeffs.shape[1]):
                H[n + 1, col] = 0.5 * coeffs[0, n] - 0.5j * coeffs[1, n]
            col += 1
        return H

    def _solve(self, n_basis: int = 401) -> np.ndarray:
        H = self._hamiltonian(n_basis)
        E = eig_banded(H, lower=True, eigvals_only=True, overwrite_a_band=True)
        E = np.asarray(E, dtype=float)
        self._energies = E - np.min(E)   # zero-point subtracted (RMG does this)
        return self._energies

    def _level_energy(self, J: int) -> float:
        if self._energies is None:
            self._solve()
        return float(self._energies[J]) if J < self._energies.shape[0] else 0.0

    def get_frequency(self) -> float:
        """RMG-Py HinderedRotor.get_frequency (torsion.pyx): the hindered-rotor
        pseudo-vibrational frequency in cm^-1. Cosine-potential form (no fourier):
        V0 = barrier/Na (J per particle);
        f = symmetry/(2*pi) * sqrt(V0/(2*I))  [1/s], returned in cm^-1."""
        if self.fourier is not None:
            raise NotImplementedError(
                "fourier-potential hindered-rotor frequency is out of scope for "
                "job-07 (no real propane_branching species use fourier data).")
        V0 = float(self.barrier) / Na
        if V0 < 0:
            raise ValueError("Hindered-rotor barrier height is less than 0.")
        freq_hz = self.symmetry / (2.0 * math.pi) * math.sqrt(V0 / (2.0 * self.inertia))
        return freq_hz / (c * 100.0)

    def get_heat_capacity(self, T: float) -> float:
        """RMG-Py HinderedRotor.get_heat_capacity (torsion.pyx:358), both
        branches. Quantum -> the eigenvalue partition function; classical +
        cosine potential (the path all real propane_branching species use,
        quantum=False semiclassical=False) -> the semiclassical-corrected Cv
        with the i0/i1 Bessel terms."""
        if self.quantum:
            if self._energies is None:
                self._solve()
            E = self._energies
            e_kT = np.exp(-E / R / T)
            Cv = (float(np.sum(E * E * e_kT)) * float(np.sum(e_kT))
                  - float(np.sum(E * e_kT)) ** 2) \
                 / (R * R * T * T * float(np.sum(e_kT)) ** 2)
            return Cv * R
        # Classical.
        if self.fourier is not None:
            raise NotImplementedError(
                "fourier-potential hindered-rotor Cv is out of scope for job-07.")
        # Cosine potential (no fourier data): the semiclassical-corrected Cv.
        from scipy.special import i0, i1
        frequency_hz = self.get_frequency() * c * 100.0   # back to Hz
        x = h * frequency_hz / (kB * T)
        z = 0.5 * float(self.barrier) / (R * T)
        exp_x = math.exp(x)
        one_minus = 1.0 - exp_x
        BB = i1(z) / i0(z)
        Cv = (x * x * exp_x / one_minus / one_minus
              - 0.5 + z * (z - BB - z * BB * BB))
        return Cv * R

    def get_sum_of_states(self, e_list, sum_states_0=None) -> np.ndarray:
        if self.quantum:
            if self._energies is None:
                self._solve()
            E = self._energies
            return convolve_bssr(e_list,
                                 np.ones_like(e_list) if sum_states_0 is None else sum_states_0,
                                 lambda J: float(E[J]) if J < E.shape[0] else float('inf'),
                                 n0=0) / self.symmetry
        # classical (cosine potential) - see RMG torsion.pyx:456-469
        if sum_states_0 is not None:
            return convolve(sum_states_0, self.get_density_of_states(e_list))
        from scipy.special import ellipe, ellipk
        q1f = math.sqrt(math.pi / self._B_energy()) / self.symmetry
        V0 = self.barrier
        pre = 4.0 * q1f * math.sqrt(V0 / (math.pi ** 3))
        out = np.zeros_like(e_list)
        for i in range(e_list.shape[0]):
            if e_list[i] < V0:
                out[i] = pre * (ellipe(e_list[i] / V0) - (1 - e_list[i] / V0) * ellipk(e_list[i] / V0))
            elif e_list[i] > V0:
                out[i] = pre * math.sqrt(e_list[i] / V0) * ellipe(V0 / e_list[i])
        return out

    def get_density_of_states(self, e_list, dens_states_0=None) -> np.ndarray:
        if self.quantum:
            if self._energies is None:
                self._solve()
            E = self._energies
            base = np.zeros_like(e_list) if dens_states_0 is None else dens_states_0
            dens = convolve_bssr(e_list, base,
                                 lambda J: float(E[J]) if J < E.shape[0] else float('inf'),
                                 n0=0) / self.symmetry
            return dens
        from scipy.special import ellipk
        dens = np.zeros_like(e_list)
        q1f = math.sqrt(math.pi / self._B_energy()) / self.symmetry
        V0 = self.barrier
        pre = 2.0 * q1f / math.sqrt(math.pi ** 3 * V0)
        for i in range(e_list.shape[0]):
            if e_list[i] < V0:
                dens[i] = pre * ellipk(e_list[i] / V0)
            elif e_list[i] > V0:
                dens[i] = pre * math.sqrt(V0 / e_list[i]) * ellipk(V0 / e_list[i])
        dens *= (e_list[1] - e_list[0])
        if dens_states_0 is not None:
            dens = convolve(dens_states_0, dens)
        return dens


class FreeRotor(Mode):
    def __init__(self, inertia: float, symmetry: int = 1, quantum: bool = False):
        super().__init__(quantum)
        self.symmetry = symmetry
        self.inertia = float(inertia)

    def get_heat_capacity(self, T: float) -> float:
        # RMG-Py FreeRotor.get_heat_capacity (torsion.pyx:669): R/2.
        return 0.5 * R

    def _A(self) -> float:
        # RMG: A = hbar / (2 I)  [1/s]  (Forst 1995 free-rotor SoS formula).
        return hbar / (2.0 * self.inertia)

    def _B_energy(self) -> float:
        return hbar * hbar / (2 * self.inertia) * Na

    def get_sum_of_states(self, e_list, sum_states_0=None) -> np.ndarray:
        # RMG-Py FreeRotor.get_sum_of_states (torsion.pyx:694, Forst 1995):
        # S(E) = (2/symmetry) * sqrt(E/A),  A = hbar/(2I).
        if sum_states_0 is not None:
            raise NotImplementedError
        A = self._A()
        return 2.0 / self.symmetry * np.sqrt(e_list / A)

    def get_density_of_states(self, e_list, dens_states_0=None) -> np.ndarray:
        # RMG-Py does NOT define a FreeRotor DoS (the .pyx ends at
        # get_sum_of_states) - the production pdep path never convolves a
        # FreeRotor DoS. Classical 1D free-rotor density kept for completeness.
        B = self._B_energy()
        dE = e_list[1] - e_list[0]
        dens = np.ones_like(e_list) * dE / B / self.symmetry
        if dens_states_0 is not None:
            dens = convolve(dens_states_0, dens)
        return dens


# Backwards-compatible alias for the step-01 test API.
class Translation(IdealGasTranslation):
    pass


# --------------------------------------------------------------------------- #
# Conformer
# --------------------------------------------------------------------------- #
class Conformer:
    def __init__(self, E0: float = 0.0, modes: Optional[List[Mode]] = None,
                 spin_multiplicity: int = 1, optical_isomers: int = 1):
        self.E0 = float(E0)
        self.modes = modes or []
        self.spin_multiplicity = spin_multiplicity
        self.optical_isomers = optical_isomers

    def get_heat_capacity(self, T: float) -> float:
        return sum(mode.get_heat_capacity(T) for mode in self.modes)

    def get_partition_function(self, T: float) -> float:
        Q = 1.0
        for mode in self.modes:
            gf = getattr(mode, "get_partition_function", None)
            if gf is not None:
                Q *= float(gf(T))
        return Q * self.spin_multiplicity * self.optical_isomers

    def get_enthalpy(self, T: float) -> float:
        """H(T) = E0 + sum over modes of the thermal enthalpy (J/mol)."""
        H = self.E0
        for mode in self.modes:
            gf = getattr(mode, "get_enthalpy", None)
            if gf is not None:
                H += float(gf(T))
        return H

    def get_entropy(self, T: float) -> float:
        """S(T) = sum over modes of the mode entropy + R*ln(spin*optical) (J/mol/K)."""
        S = 0.0
        for mode in self.modes:
            gf = getattr(mode, "get_entropy", None)
            if gf is not None:
                S += float(gf(T))
        S += R * math.log(self.spin_multiplicity * self.optical_isomers)
        return S

    def get_free_energy(self, T: float) -> float:
        """G(T) = H(T) - T*S(T) (J/mol). Used for network equilibrium ratios."""
        return self.get_enthalpy(T) - T * self.get_entropy(T)

    def get_sum_of_states(self, e_list) -> np.ndarray:
        sum_states = None
        for mode in self.modes:
            sum_states = mode.get_sum_of_states(e_list, sum_states)
        if sum_states is None:
            sum_states = np.ones_like(e_list)
        return sum_states * self.spin_multiplicity * self.optical_isomers

    def get_density_of_states(self, e_list) -> np.ndarray:
        dens_states = None
        for mode in self.modes:
            dens_states = mode.get_density_of_states(e_list, dens_states)
        if dens_states is None:
            dens_states = np.zeros_like(e_list)
            dens_states[0] = 1.0
        return dens_states * self.spin_multiplicity * self.optical_isomers


# --------------------------------------------------------------------------- #
# Conformer reconstruction from a serializable spec (the reference / gate
# seam). A spec is a dict with "E0", "spin_multiplicity", "optical_isomers"
# and a "modes" list of {type, ...} entries (the same shape rmgpy_pdep_
# reference.py dumps). Units: frequencies cm^-1, inertia kg m^2, barrier /
# E0 J/mol, mass kg.
# --------------------------------------------------------------------------- #
def conformer_from_spec(spec: dict) -> Conformer:
    modes: List[Mode] = []
    for m in spec.get("modes", []):
        t = m["type"]
        if t == "HarmonicOscillator":
            modes.append(HarmonicOscillator(frequencies=list(m["frequencies_cm1"]),
                                            quantum=True))
        elif t == "IdealGasTranslation":
            modes.append(IdealGasTranslation(mass=m["mass"], quantum=False))
        elif t == "NonlinearRotor":
            modes.append(NonlinearRotor(inertia=list(m["inertia"]),
                                        symmetry=m.get("symmetry", 1),
                                        quantum=False))
        elif t == "LinearRotor":
            modes.append(LinearRotor(inertia=m.get("inertia"),
                                     symmetry=m.get("symmetry", 1),
                                     quantum=True))
        elif t == "HinderedRotor":
            modes.append(HinderedRotor(inertia=m.get("inertia"),
                                       symmetry=m.get("symmetry", 1),
                                       barrier=m.get("barrier"),
                                       fourier=m.get("fourier"),
                                       quantum=bool(m.get("quantum", False))))
        elif t == "FreeRotor":
            modes.append(FreeRotor(inertia=m.get("inertia"),
                                   symmetry=m.get("symmetry", 1)))
    return Conformer(E0=float(spec.get("E0", 0.0) or 0.0),
                     modes=modes,
                     spin_multiplicity=int(spec.get("spin_multiplicity", 1)),
                     optical_isomers=int(spec.get("optical_isomers", 1)))
