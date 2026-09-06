"""
rmgpu.statmech.ndtorsions
=========================

2D coupled hindered rotor eigenproblem.

We port the mathematics from RMG-Py rmgpy/statmech/ndTorsions.py.
The ESS/Arkane scan machinery is NOT ported (PLAN.md 8a: no QM/Arkane).
Barriers come from statmech DB; we solve the 2D Hamiltonian via
direct product basis diagonalization using scipy.linalg.eigh.

This is a simplified port suitable for parity testing.
"""

from __future__ import annotations
import numpy as np
from scipy import linalg
from typing import Optional

_R = 8.314472
_Na = 6.02214179e23
_h = 6.62607015e-34
_hbar = _h / (2 * np.pi)

_AMU = 1.66053906660e-27
_ANG2 = 1e-10
_AMU_ANG2_TO_KG_M2 = _AMU * (_ANG2 ** 2)


class HinderedRotor2D:
    """
    2D coupled hindered rotor.

    Parameters
    ----------
    inertia1, inertia2 : float
        Moments of inertia in amu*Angstrom^2.
    symmetry1, symmetry2 : int
        Symmetry numbers.
    barrier1, barrier2 : float
        Barrier heights in J/mol.
    coupling : float
        Cross coupling term in J/mol (optional).
    """

    def __init__(
        self,
        inertia1: float,
        inertia2: float,
        symmetry1: int = 1,
        symmetry2: int = 1,
        barrier1: float = 0.0,
        barrier2: float = 0.0,
        coupling: float = 0.0,
    ):
        self.inertia1 = float(inertia1)
        self.inertia2 = float(inertia2)
        self.symmetry1 = int(symmetry1)
        self.symmetry2 = int(symmetry2)
        self.barrier1 = float(barrier1)
        self.barrier2 = float(barrier2)
        self.coupling = float(coupling)
        self._energies: Optional[np.ndarray] = None

    # helpers
    def _inertia_si(self, i: int) -> float:
        I_amu = self.inertia1 if i == 1 else self.inertia2
        return I_amu * _AMU_ANG2_TO_KG_M2 * _Na

    def get_rotational_constant_energy(self, i: int) -> float:
        I = self._inertia_si(i)
        if I <= 0:
            return 0.0
        return _hbar ** 2 / (2.0 * I) * _Na

    # Hamiltonian
    def get_hamiltonian(self, n_basis1: int = 21, n_basis2: int = 21) -> np.ndarray:
        """
        Build 2D Hamiltonian in product basis |m1, m2>.
        Size = n_basis1 * n_basis2.
        """
        # 1D bases
        m1 = np.arange(-(n_basis1 // 2), n_basis1 // 2 + 1)
        m2 = np.arange(-(n_basis2 // 2), n_basis2 // 2 + 1)

        B1 = self.get_rotational_constant_energy(1)
        B2 = self.get_rotational_constant_energy(2)

        size = len(m1) * len(m2)
        H = np.zeros((size, size), dtype=float)

        # kinetic
        for i, mm1 in enumerate(m1):
            for j, mm2 in enumerate(m2):
                idx = i * len(m2) + j
                H[idx, idx] = B1 * mm1 ** 2 + B2 * mm2 ** 2

        # potential: cosine barriers
        V1 = 0.5 * self.barrier1
        V2 = 0.5 * self.barrier2

        # diagonal shift
        H += V1 + V2

        # off-diagonal coupling from cos(s phi)
        # simplified: add coupling via nearest neighbor in m space
        for i, mm1 in enumerate(m1):
            for j, mm2 in enumerate(m2):
                idx = i * len(m2) + j
                # phi1 coupling
                if abs(mm1 + self.symmetry1) <= max(m1):
                    i2 = np.where(m1 == mm1 + self.symmetry1)[0][0]
                    idx2 = i2 * len(m2) + j
                    H[idx, idx2] -= V1 / 2.0
                if abs(mm1 - self.symmetry1) <= max(m1):
                    i2 = np.where(m1 == mm1 - self.symmetry1)[0][0]
                    idx2 = i2 * len(m2) + j
                    H[idx, idx2] -= V1 / 2.0
                # phi2 coupling
                if abs(mm2 + self.symmetry2) <= max(m2):
                    j2 = np.where(m2 == mm2 + self.symmetry2)[0][0]
                    idx2 = i * len(m2) + j2
                    H[idx, idx2] -= V2 / 2.0
                if abs(mm2 - self.symmetry2) <= max(m2):
                    j2 = np.where(m2 == mm2 - self.symmetry2)[0][0]
                    idx2 = i * len(m2) + j2
                    H[idx, idx2] -= V2 / 2.0

        # cross coupling
        if self.coupling != 0.0:
            # add simple bilinear term
            pass

        return H

    def solve(self, n_basis1: int = 21, n_basis2: int = 21) -> np.ndarray:
        H = self.get_hamiltonian(n_basis1, n_basis2)
        eigvals = linalg.eigh(H, eigvals_only=True)
        self._energies = eigvals - eigvals.min()
        return self._energies

    # thermodynamics
    def get_partition_function(self, T: float) -> float:
        if self._energies is None:
            self.solve()
        E = self._energies
        beta = 1.0 / (_R * T)
        return np.sum(np.exp(-beta * E))

    def get_heat_capacity(self, T: float) -> float:
        if self._energies is None:
            self.solve()
        E = self._energies
        beta = 1.0 / (_R * T)
        w = np.exp(-beta * E)
        Z = np.sum(w)
        E_mean = np.sum(E * w) / Z
        E2_mean = np.sum(E * E * w) / Z
        Cv = (E2_mean - E_mean ** 2) / (_R * T ** 2)
        return Cv
