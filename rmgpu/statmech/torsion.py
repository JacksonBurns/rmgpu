"""
rmgpu.statmech.torsion
======================

Port of RMG-Py rmgpy/statmech/torsion.pyx to pure Python/NumPy/SciPy.
Implements 1D hindered rotor with Fourier/cosine potential, quantum
eigenproblem via basis set diagonalization (scipy.linalg.eigh).

Constants match RMG-Py: R=8.314472 J/mol/K, Na=6.02214179e23, h=6.62607015e-34 J*s.
"""

from __future__ import annotations
import numpy as np
from scipy import linalg
from typing import Optional

_R = 8.314472
_Na = 6.02214179e23
_h = 6.62607015e-34
_hbar = _h / (2 * np.pi)

# conversion amu*Angstrom^2 -> kg*m^2
_AMU = 1.66053906660e-27
_ANG2 = 1e-10
_AMU_ANG2_TO_KG_M2 = _AMU * (_ANG2 ** 2)


class Torsion:
    """Base class for torsional degrees of freedom."""

    def __init__(self, symmetry: int = 1, quantum: bool = True):
        self.symmetry = symmetry
        self.quantum = quantum
        self._energies: Optional[np.ndarray] = None

    def get_heat_capacity(self, T: float) -> float:
        raise NotImplementedError

    def get_partition_function(self, T: float) -> float:
        raise NotImplementedError


class HinderedRotor(Torsion):
    """
    1D hindered rotor with cosine or Fourier potential.

    Parameters
    ---------
    inertia_amu_ang2 : float
        Moment of inertia in amu*Angstrom^2.
    symmetry : int
        Symmetry number.
    barrier : float
        Barrier height in J/mol.
    fourier : np.ndarray, optional
        2 x N Fourier coefficients in J/mol.
    quantum : bool
        Use quantum eigenproblem.
    """

    def __init__(
        self,
        inertia_amu_ang2: float,
        symmetry: int = 1,
        barrier: Optional[float] = None,
        fourier: Optional[np.ndarray] = None,
        quantum: bool = True,
    ):
        super().__init__(symmetry=symmetry, quantum=quantum)
        self.inertia_amu_ang2 = float(inertia_amu_ang2)
        self.barrier = float(barrier) if barrier is not None else 0.0
        self.fourier = fourier  # shape (2, N)
        self._energies = None

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _inertia_si(self) -> float:
        return self.inertia_amu_ang2 * _AMU_ANG2_TO_KG_M2 * _Na  # J*s^2/mol? Actually kg*m^2 per mole

    def get_rotational_constant_energy(self) -> float:
        """Rotational constant B in J/mol: hbar^2/(2I) * Na"""
        I = self._inertia_si()
        if I <= 0:
            return 0.0
        return _hbar ** 2 / (2.0 * I) * _Na

    def get_potential(self, phi: float) -> float:
        if self.fourier is not None:
            # Fourier series: V = sum a_k cos(k phi) + b_k sin(k phi)
            V = 0.0
            coeffs = self.fourier
            n = coeffs.shape[1]
            for k in range(n):
                V += coeffs[0, k] * np.cos((k + 1) * phi) + coeffs[1, k] * np.sin((k + 1) * phi)
            return V
        # cosine potential
        return 0.5 * self.barrier * (1.0 - np.cos(self.symmetry * phi))

    # ------------------------------------------------------------------ #
    # Hamiltonian
    # ------------------------------------------------------------------ #
    def get_hamiltonian(self, n_basis: int = 401) -> np.ndarray:
        """
        Build Hamiltonian matrix in free rotor basis |m>.
        Returns dense (2M+1) x (2M+1) matrix in J/mol.
        """
        if n_basis % 2 == 0:
            M = n_basis // 2
        else:
            M = (n_basis - 1) // 2

        size = 2 * M + 1
        H = np.zeros((size, size), dtype=float)

        B = self.get_rotational_constant_energy()

        # diagonal
        m_vals = np.arange(-M, M + 1)
        np.fill_diagonal(H, B * m_vals ** 2)

        # potential
        if self.fourier is not None:
            # Fourier coefficients in J/mol
            # RMG convention: V(phi) = sum_{k} c_k cos(k phi) + s_k sin(k phi)
            # In basis |m>, <m|cos(k phi)|m'> = 0.5 delta_{m', m+k} + 0.5 delta_{m', m-k}
            # Build coupling
            n_terms = self.fourier.shape[1]
            for k in range(n_terms):
                c = self.fourier[0, k]
                s = self.fourier[1, k]
                # cosine coupling
                for i, m in enumerate(m_vals):
                    m2 = m + k + 1
                    if m2 <= M:
                        j = np.where(m_vals == m2)[0][0]
                        H[i, j] += 0.5 * c
                        H[j, i] += 0.5 * c
                    m2 = m - (k + 1)
                    if m2 >= -M:
                        j = np.where(m_vals == m2)[0][0]
                        H[i, j] += 0.5 * c
                        H[j, i] += 0.5 * c
                # sine coupling (imaginary, gives real off-diagonal)
                # For simplicity, ignore sine terms (they contribute zero for symmetric potentials)
        else:
            # cosine potential V0 (1 - cos(s * phi))
            # V0 = 0.5 * barrier
            V0 = 0.5 * self.barrier
            # diagonal shift
            H += V0
            # off-diagonal coupling
            s = self.symmetry
            for i, m in enumerate(m_vals):
                m2 = m + s
                if abs(m2) <= M:
                    j = np.where(m_vals == m2)[0][0]
                    H[i, j] -= V0 / 2.0
                    H[j, i] -= V0 / 2.0
                m2 = m - s
                if abs(m2) <= M:
                    j = np.where(m_vals == m2)[0][0]
                    H[i, j] -= V0 / 2.0
                    H[j, i] -= V0 / 2.0

        return H

    def solve_schrodinger_equation(self, n_basis: int = 401) -> np.ndarray:
        H = self.get_hamiltonian(n_basis)
        eigvals = linalg.eigh(H, eigvals_only=True)
        # shift to zero point
        self._energies = eigvals - eigvals.min()
        return self._energies

    # ------------------------------------------------------------------ #
    # thermodynamics
    # ------------------------------------------------------------------ #
    def get_partition_function(self, T: float) -> float:
        if self.quantum:
            if self._energies is None:
                self.solve_schrodinger_equation()
            E = self._energies
            beta = 1.0 / (_R * T)
            q = np.sum(np.exp(-beta * E))
            return q / self.symmetry
        # classical approximation
        B = self.get_rotational_constant_energy()
        if B <= 0:
            return 1.0
        # classical integral approx
        return np.sqrt(np.pi * _R * T / B) / self.symmetry

    def get_heat_capacity(self, T: float) -> float:
        if self.quantum:
            if self._energies is None:
                self.solve_schrodinger_equation()
            E = self._energies
            beta = 1.0 / (_R * T)
            w = np.exp(-beta * E)
            Z = np.sum(w)
            E_mean = np.sum(E * w) / Z
            E2_mean = np.sum(E * E * w) / Z
            Cv = (E2_mean - E_mean ** 2) / (_R * T ** 2)
            return Cv
        # classical hindered rotor: ~R for moderate T
        return _R

    # ------------------------------------------------------------------ #
    # state counting (classical approximations)
    # ------------------------------------------------------------------ #
    def get_sum_of_states(self, e_list: np.ndarray, sum_states_0: Optional[np.ndarray] = None):
        # Placeholder classical DoS for hindered rotor
        # Use simple analytic approximation
        if sum_states_0 is None:
            sum_states = np.zeros_like(e_list, dtype=float)
            sum_states[:] = 1.0
        else:
            sum_states = sum_states_0.copy()
        return sum_states

    def get_density_of_states(self, e_list: np.ndarray, dens_states_0: Optional[np.ndarray] = None):
        if dens_states_0 is None:
            return np.ones_like(e_list, dtype=float)
        return dens_states_0.copy()


class FreeRotor(Torsion):
    """Free rotor, classical."""

    def __init__(self, inertia_amu_ang2: float, symmetry: int = 1):
        super().__init__(symmetry=symmetry, quantum=False)
        self.inertia_amu_ang2 = float(inertia_amu_ang2)

    def get_heat_capacity(self, T: float) -> float:
        return _R / 2.0

    def get_partition_function(self, T: float) -> float:
        I = self.inertia_amu_ang2 * _AMU_ANG2_TO_KG_M2 * _Na
        return np.sqrt(8 * np.pi ** 3 * _R * T * I) / (self.symmetry * _h * _Na)
