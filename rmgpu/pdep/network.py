"""
rmgpu.pdep.network
==================

job-07/step-04: the pressure-dependent unimolecular reaction network + the
master-equation (ME) machinery for the CSE (chemically-significant eigenvalues,
Allen) k(T,P) extraction, and the no-QM transition-state E0 derivation.

Ported from RMG-Py (rmgpy/pdep/network.py, me.pyx, cse.pyx, reaction.pyx,
rmgpy/rmg/pdep.py). SI units throughout (J, K, Pa, mol). No QM, no Cython.

Scope (this step, step-04):
  - Grain generation (select_energy_grains / _get_energy_grains) -- ported.
  - Equilibrium ratios (calculate_equilibrium_ratios) -- ported.
  - The no-QM TS E0 derivation (derive_ts_e0) -- ported exactly from
    rmgpy/rmg/pdep.py:856 (Ea-based; see the note on the brief's TST form).
  - The full ME matrix (generate_full_me_matrix) -- ported from me.pyx.
  - The CSE (Allen) k(T,P) extraction (apply_cse / calculate_rate_coefficients)
    -- ported from cse.pyx.
  - The ILT k(E) path (apply_ilt_k_e) -- ported from reaction.pyx (the no-QM
    microcanonical rate, used by the core loop for pdep reactions whose TS has
    no vibrational data, only an E0).

Out of scope (a separate step):
  - The collision matrix / collision frequency (SingleExponentialDown, CSE
    collision) -- step-05 (rmgpu/pdep/collision.py). The Network exposes a
    *collision protocol* (seed the collision matrix + collision frequency) so
    step-05 plugs in without touching this module.
  - The DoS (step-01/03). The Network accepts the DoS as an input (seeded in
    the unit test from the RMG-Py reference); it does not compute it.

The Network is *state-seeded* for the step-04 unit test: it consumes the
per-(T,P) state (e_list, DoS, collision matrix, k(E) rate matrices,
equilibrium ratios) recorded by an independent RMG-Py reference
(scripts/record_job07_step04_reference.py, run in rmg_env ->
gates/baselines/job07/toy_lindemann_ref.json) and reproduces RMG-Py's CSE
k(T,P). This is non-circular: the reference is RMG-Py's own run, never rmgpu.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Constants: matched to RMG-Py (rmgpy/constants.py) EXACTLY. A mismatch here
# breaks the 1e-8 no-QM TS-E0 parity gate (PLAN 8a.2) and the k(T,P) parity.
# --------------------------------------------------------------------------- #
R = 8.314472                 # J/mol/K   (RMG-Py value, NOT CODATA)
h = 6.62606896e-34           # J*s
Na = 6.02214179e23           # 1/mol
c = 299792458.0              # m/s
# RMG-Py hardcodes kB = 1.3806504e-23 (rmgpy/constants.py), which differs from
# R/Na = 1.38065032e-23 by 5.7e-8. The collision frequency scales as
# kB**(-1/2), so matching RMG's exact kB is required for the 1e-9 frequency
# parity (the grain tail 40*kB*T is unaffected at this precision).
kB = 1.3806504e-23
hbar = h / (2.0 * np.pi)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class PDepNetworkError(Exception):
    """A pressure-dependent network construction / solve error."""


# --------------------------------------------------------------------------- #
# The no-QM TS E0 derivation (RMG-Py rmgpy/rmg/pdep.py:856, arkane/pdep.py:280)
# --------------------------------------------------------------------------- #
def derive_ts_e0(
    reactant_e0s: Sequence[float],
    e_a: float,
    energy_correction: float = 0.0,
) -> float:
    """
    Derive the transition-state ground-state energy E0_TS from the
    reactant ground-state energies and the high-pressure-limit activation
    energy, with NO QM. This is RMG-Py's actual no-QM path (the core loop
    builds the TS conformer with an E0 only, no vibrational modes):

        E0_TS = sum(reactant E0) + Ea + energy_correction

    Parameters
    ----------
    reactant_e0s : sequence of float
        Ground-state energies (J/mol) of the path-reaction reactants.
    e_a : float
        High-pressure-limit activation energy (J/mol) of the reaction.
    energy_correction : float, optional
        The network reference-energy correction (J/mol),
        = -min over the network's stationary-point channel E0s
        (rmgpy/rmg/pdep.py:823). Zero when energies are already referenced.

    Returns
    -------
    float
        The TS ground-state energy E0_TS in J/mol.

    Note
    ----
    The step brief cites the TST form E0_TS = sum(reactant E0) - R*T*ln(k_inf*V/h).
    RMG-Py implements the Ea-based form above (the two are equivalent at the
    HPL limit up to the T^n and tunneling corrections, which RMG-Py folds into
    the Arrhenius fit). Because the job-07 gate compares E0_TS to RMG-Py's
    value within 1e-8 (a BLOCKER per PLAN 8a.2), this function ports RMG-Py's
    exact expression. The k_inf*V/h form is NOT what RMG-Py's core loop uses.
    """
    return float(sum(reactant_e0s)) + float(e_a) + float(energy_correction)


def network_energy_correction(
    channel_e0s: Sequence[float],
) -> float:
    """
    The network reference-energy correction (rmgpy/rmg/pdep.py:823):
    energy_correction = -min over the network's stationary-point channel E0s
    (isomer/reactant/product channel total ground-state energies, J/mol).
    """
    return -float(min(channel_e0s))


# --------------------------------------------------------------------------- #
# The Network
# --------------------------------------------------------------------------- #
class Network:
    """
    A pressure-dependent unimolecular reaction network.

    This is a *state-seeded* network: it consumes the per-condition state
    (grain grid, DoS, collision matrix, k(E) rate matrices, equilibrium
    ratios) that RMG-Py's own run records (non-circular reference), and
    reproduces RMG-Py's CSE k(T,P). The parts this step OWNS and computes
    are: grain generation, the ME matrix, the CSE k(T,P) extraction, the
    ILT k(E) path, and the no-QM TS E0. The collision model and the DoS are
    seeded (they are step-05's and step-01/03's deliverables respectively).

    Attributes
    ----------
    label : str
    e_list : np.ndarray (n_grains,)         energy grains, J/mol
    j_list : np.ndarray (n_j,)              angular-momentum grains (int)
    dens_states : np.ndarray (n_cfg, n_grains, n_j)   rescaled DoS (Q=1)
    eq_ratios : np.ndarray (n_cfg,)         equilibrium ratios (isomers, channels)
    Kij, Gnj, Fim : np.ndarray (..., n_grains, n_j)   (rescaled) k(E) matrices
    coll_freq : np.ndarray (n_isom,)        collision frequency, s^-1
    Mcoll : np.ndarray (n_isom, n_grains, n_j, n_grains, n_j)   collision matrix
    K : np.ndarray (n_cfg, n_cfg)           phenomenological k(T,P) at the current T,P
    """

    def __init__(
        self,
        label: str = "",
        n_isom: int = 1,
        n_reac: int = 0,
        n_prod: int = 1,
        e_list: Optional[np.ndarray] = None,
        j_list: Optional[np.ndarray] = None,
        dens_states: Optional[np.ndarray] = None,
        eq_ratios: Optional[np.ndarray] = None,
        Kij: Optional[np.ndarray] = None,
        Gnj: Optional[np.ndarray] = None,
        Fim: Optional[np.ndarray] = None,
        coll_freq: Optional[np.ndarray] = None,
        Mcoll: Optional[np.ndarray] = None,
        E0: Optional[np.ndarray] = None,
        E0_ts: Optional[np.ndarray] = None,
        Tmax: Optional[float] = None,
        Tmin: Optional[float] = None,
        Pmin: Optional[float] = None,
        Pmax: Optional[float] = None,
        grain_size: float = 0.0,
        grain_count: int = 0,
        active_j_rotor: bool = True,
    ):
        self.label = label
        self.n_isom = int(n_isom)
        self.n_reac = int(n_reac)
        self.n_prod = int(n_prod)
        self.n_cfg = self.n_isom + self.n_reac + self.n_prod
        self.e_list = np.asarray(e_list, dtype=float) if e_list is not None else None
        self.j_list = np.asarray(j_list, dtype=int) if j_list is not None else np.array([0], dtype=int)
        self.dens_states = (np.asarray(dens_states, dtype=float) if dens_states is not None else None)
        self.eq_ratios = (np.asarray(eq_ratios, dtype=float) if eq_ratios is not None else None)
        self.Kij = (np.asarray(Kij, dtype=float) if Kij is not None else None)
        self.Gnj = (np.asarray(Gnj, dtype=float) if Gnj is not None else None)
        self.Fim = (np.asarray(Fim, dtype=float) if Fim is not None else None)
        self.coll_freq = (np.asarray(coll_freq, dtype=float) if coll_freq is not None else None)
        self.Mcoll = (np.asarray(Mcoll, dtype=float) if Mcoll is not None else None)
        self.E0 = (np.asarray(E0, dtype=float) if E0 is not None else None)
        self.E0_ts = (np.asarray(E0_ts, dtype=float) if E0_ts is not None else None)
        self.Tmax = float(Tmax) if Tmax is not None else None
        self.Tmin = float(Tmin) if Tmin is not None else None
        self.Pmin = float(Pmin) if Pmin is not None else None
        self.Pmax = float(Pmax) if Pmax is not None else None
        self.grain_size = float(grain_size)
        self.grain_count = int(grain_count)
        self.active_j_rotor = bool(active_j_rotor)
        self.T = 0.0
        self.P = 0.0
        self.K = None

    # ------------------------------------------------------------------ #
    # Collision protocol (step-05 plugs in here; seeded in the unit test)
    # ------------------------------------------------------------------ #
    def seed_collision(self, coll_freq: np.ndarray, Mcoll: np.ndarray) -> None:
        """
        Set the collision frequency and collision matrix. Step-05's collision
        model will compute these from the bath gas + energy-transfer model;
        for the step-04 unit test they are seeded from the RMG-Py reference.
        """
        self.coll_freq = np.asarray(coll_freq, dtype=float)
        self.Mcoll = np.asarray(Mcoll, dtype=float)

    # ------------------------------------------------------------------ #
    # Grain generation (ported from rmgpy/pdep/network.py)
    # ------------------------------------------------------------------ #
    def _get_energy_grains(self, Emin: float, Emax: float,
                           grain_size: float = 0.0, grain_count: int = 0) -> np.ndarray:
        if grain_count <= 0 and grain_size <= 0.0:
            raise PDepNetworkError(
                "You must specify a positive value for either dE or n_grains.")
        elif grain_count <= 0 and grain_size > 0.0:
            use_grain_size = True
        elif grain_count > 0 and grain_size <= 0.0:
            use_grain_size = False
        else:
            grain_size0 = (Emax - Emin) / (grain_count - 1)
            use_grain_size = (grain_size0 > grain_size)
        if use_grain_size:
            return np.arange(Emin, Emax + grain_size, grain_size, dtype=float)
        else:
            return np.linspace(Emin, Emax, grain_count, dtype=float)

    def select_energy_grains(self, T: float,
                             grain_size: float = 0.0, grain_count: int = 0) -> np.ndarray:
        """
        Select the energy grains at temperature T. Ported from
        rmgpy/pdep/network.py select_energy_grains.

        e_min = floor(min channel E0); e_max = max(max channel E0, TS E0s)
        + 40*kB*T (RMG adds the transition-state energy and the +40kT tail).
        """
        if grain_size == 0.0 and grain_count == 0:
            raise PDepNetworkError(
                "Must provide either grain_size or n_grains to select_energy_grains.")
        if self.E0 is None:
            raise PDepNetworkError("E0 is not set; cannot select energy grains.")
        e_min = float(min(self.E0))
        e_min = np.floor(e_min)  # round down to the nearest whole J/mol (RMG)
        e_max = float(max(self.E0))
        # Include the transition-state ground-state energies (RMG network.py:
        # e_max is the highest energy on the PES, including the TS).
        if self.E0_ts is not None:
            e_max = max(e_max, float(max(self.E0_ts)))
        e_max += 40.0 * R * T
        return self._get_energy_grains(e_min, e_max, grain_size, grain_count)

    # ------------------------------------------------------------------ #
    # Equilibrium ratios (ported from rmgpy/pdep/network.py)
    # ------------------------------------------------------------------ #
    def calculate_equilibrium_ratios(self, G: np.ndarray, T: float) -> np.ndarray:
        """
        Equilibrium ratios from the Gibbs free energy (J/mol) of each
        isomer/channel. Ported from rmgpy/pdep/network.py. For the isomer
        (the only configuration with statmech in the toy unimolecular
        network): eq_ratios[i] = exp(-G_i / R / T).
        """
        conc = 1.0e5 / R / T
        n = self.n_cfg
        eq = np.zeros(n, float)
        for i in range(self.n_isom):
            eq[i] = np.exp(-G[i] / R / T)
        # reactant/product channels: exp(-G/R/T) * conc**(n_species - 1);
        # for the toy unimolecular network the product channel has no
        # statmech, so RMG leaves it at 0 (mirrors network.py guard).
        return eq

    # ------------------------------------------------------------------ #
    # The full master equation matrix (ported from rmgpy/pdep/me.pyx)
    # ------------------------------------------------------------------ #
    def generate_full_me_matrix(self, products: bool = True,
                                neglect_high_energy_collisions: bool = False,
                                high_energy_rate_tol: float = 0.01):
        """
        Assemble the full master-equation matrix + the accounting matrix
        `indices`. Ported line-for-line from rmgpy/pdep/me.pyx.
        `indices` maps (isomer, grain, j) -> ME row index (-1 if absent);
        the reactant/product channels occupy the trailing rows.
        """
        e_list = self.e_list
        j_list = self.j_list
        dens_states = self.dens_states
        m_coll = self.Mcoll
        k_ij = self.Kij
        f_im = self.Fim
        g_nj = self.Gnj
        n_isom = self.n_isom
        n_reac = self.n_reac
        n_prod = self.n_prod
        n_grains = self.e_list.shape[0]
        n_j = self.j_list.shape[0]
        T = self.T
        beta = 1.0 / (R * T)

        # Accounting matrix
        indices = -np.ones((n_isom, n_grains, n_j), int)
        n_rows = 0
        for r in range(n_grains):
            for s in range(n_j):
                for i in range(n_isom):
                    if dens_states[i, r, s] > 0:
                        indices[i, r, s] = n_rows
                        n_rows += 1
        n_rows += n_reac
        if products:
            n_rows += n_prod

        me = np.zeros((n_rows, n_rows), float)

        # Collision terms
        for i in range(n_isom):
            for r in range(n_grains):
                for s in range(n_j):
                    if indices[i, r, s] > -1:
                        for u in range(r, n_grains):
                            for v in range(s, n_j):
                                me[indices[i, r, s], indices[i, u, v]] = m_coll[i, r, s, u, v]
                                me[indices[i, u, v], indices[i, r, s]] = m_coll[i, u, v, r, s]

        # Isomerization terms
        for i in range(n_isom):
            for j in range(i):
                if k_ij[i, j, n_grains - 1, 0] > 0 or k_ij[j, i, n_grains - 1, 0] > 0:
                    for r in range(n_grains):
                        for s in range(n_j):
                            u, v = indices[i, r, s], indices[j, r, s]
                            if u > -1 and v > -1:
                                me[v, u] = k_ij[j, i, r, s]
                                me[u, u] -= k_ij[j, i, r, s]
                                me[u, v] = k_ij[i, j, r, s]
                                me[v, v] -= k_ij[i, j, r, s]

        # Association / dissociation terms
        for i in range(n_isom):
            for n in range(n_reac + n_prod):
                if g_nj[n, i, n_grains - 1, 0] > 0:
                    for r in range(n_grains):
                        for s in range(n_j):
                            u = indices[i, r, s]
                            v = n_rows - n_reac - n_prod + n if products else n_rows - n_reac + n
                            if u > -1:
                                me[u, u] -= g_nj[n, i, r, s]
                                if n < n_reac or products:
                                    me[v, u] = g_nj[n, i, r, s]
                                if n < n_reac:
                                    val = f_im[i, n, r, s] * dens_states[n + n_isom, r, s] \
                                          * (2 * j_list[s] + 1) * np.exp(-e_list[r] * beta)
                                    me[u, v] = val
                                    me[v, v] -= val

        # (neglect_high_energy_collisions: omitted - RMG default is False and
        #  the toy network has a single isomer; step-07/08 may add it.)
        if not products:
            return me, indices
        return me, indices

    # ------------------------------------------------------------------ #
    # The CSE (Allen) k(T,P) extraction (ported from rmgpy/pdep/cse.pyx)
    # ------------------------------------------------------------------ #
    def apply_cse_allen(self) -> np.ndarray:
        """
        Compute the phenomenological k(T,P) matrix at the current (T, P) using
        the CSE (Allen) method. Ported from rmgpy/pdep/cse.pyx
        apply_chemically_significant_eigenvalues_method (method='allen').
        Returns the (n_cfg, n_cfg) k(T,P) matrix.
        """
        if self.dens_states is None or self.eq_ratios is None:
            raise PDepNetworkError("dens_states and eq_ratios must be seeded before apply_cse_allen.")
        T = self.T
        P = self.P
        e_list = self.e_list
        j_list = self.j_list
        dens_states = self.dens_states
        g_nj = self.Gnj
        eq_ratios = self.eq_ratios
        n_isom = self.n_isom
        n_reac = self.n_reac
        n_prod = self.n_prod
        n_cfg = self.n_cfg
        n_grains = len(e_list)
        n_chem = n_isom + n_reac
        ym_b = 1.0e-6 * P / (R * T)

        me_mat, indices = self.generate_full_me_matrix(products=False)
        n_rows = me_mat.shape[0]
        me_mat[:, n_rows - n_reac:] *= ym_b

        # Symmetrization
        s_mat = np.zeros(n_rows, float)
        s_mat_inv = np.zeros_like(s_mat)
        for i in range(n_isom):
            for r in range(n_grains):
                for s in range(n_j := len(j_list)):
                    idx = indices[i, r, s]
                    if idx > -1:
                        s_mat[idx] = np.sqrt(
                            dens_states[i, r, s] * (2 * j_list[s] + 1)
                            * np.exp(-e_list[r] / (R * T)) * eq_ratios[i])
                        s_mat_inv[idx] = 1.0 / s_mat[idx]
        for n in range(n_reac):
            idx = n_rows - n_reac + n
            s_mat[idx] = np.sqrt(eq_ratios[n + n_isom] / ym_b)
            s_mat_inv[idx] = 1.0 / s_mat[idx]

        for r in range(n_rows):
            for s in range(n_rows):
                if s_mat_inv[r] != 0.0 and s_mat[s] != 0.0:
                    me_mat[r, s] = s_mat_inv[r] * me_mat[r, s] * s_mat[s]

        # Symmetrization check (RMG cse.pyx): a non-symmetrized matrix is a
        # hard error, not a silent no-op.
        for r in range(n_rows):
            for s in range(r):
                if me_mat[r, s] != 0:
                    if abs(me_mat[r, s] - me_mat[s, r]) > 0.01 * min(me_mat[r, s], me_mat[s, r]) \
                            and max(me_mat[r, s], me_mat[s, r]) > 1e-200:
                        raise PDepNetworkError("Master equation matrix not properly symmetrized.")

        # Eigen-decomposition (RMG uses a full eigh for the Allen method)
        omega0, eig0 = np.linalg.eigh(me_mat)
        ind = np.argsort(omega0)

        # Count distinct (chemically-significant) eigenvalues (RMG cse.pyx).
        n_cse = 0
        for i in range(n_chem):
            if abs(omega0[ind[-n_chem - 1]] / omega0[ind[-1 - i]]) > 3.0:
                n_cse += 1
        if n_cse != n_chem:
            # RMG (cse.pyx): no lumping order is passed from network.py, so it
            # CANNOT recover - it logs and returns the zero k(T,P) matrix.
            # Match that exactly (a silent no-op would hide a real failure).
            print("[rmgpu.pdep] CSE: could only identify %d distinct eigenvalues, "
                  "when %d are required; returning zero k(T,P)." % (n_cse, n_chem))
            self.K = np.zeros((n_cfg, n_cfg), float)
            return self.K

        k = np.zeros((n_cfg, n_cfg), float)
        omega = omega0.take(ind[-n_cse:])
        eig = eig0.take(ind[-n_cse:], axis=1)
        for j in range(n_cse):
            eig[:, j] *= s_mat

        z_mat = np.zeros((n_cse, n_cse), float)
        z_inv = np.zeros((n_cse, n_cse), float)
        y_mat = np.zeros((n_prod, n_cse), float)
        for j in range(n_cse):
            for i in range(n_isom):
                for r in range(n_grains):
                    for s in range(len(j_list)):
                        idx = indices[i, r, s]
                        if idx > -1:
                            z_mat[i, j] += eig[idx, j]
        for j in range(n_cse):
            for i in range(n_isom):
                for r in range(n_grains):
                    for s in range(len(j_list)):
                        idx = indices[i, r, s]
                        if idx > -1:
                            for n in range(n_prod):
                                y_mat[n, j] += g_nj[n_reac + n, i, r, s] * eig[idx, j]
        for j in range(n_cse):
            for n in range(n_reac):
                idx = n_rows - n_reac + n
                z_mat[n_isom + n, j] += eig[idx, j]

        z_inv = np.linalg.inv(z_mat)

        # K over the chemical (isomer+reactant) block
        for i in range(n_chem):
            for j in range(n_chem):
                k[i, j] = np.sum(z_mat[i, :] * omega * z_inv[:, j])
        # Product rows (forward association/dissociation flux)
        for n in range(n_prod):
            for j in range(n_chem):
                k[n_isom + n_reac + n, j] = np.sum(y_mat[n, :] * z_inv[:, j])

        # Convert the product/reactant entries from the 1e-6*P scaling back
        k[:, n_isom:] /= ym_b
        self.K = k
        return k

    # ------------------------------------------------------------------ #
    # The ILT k(E) path (ported from rmgpy/pdep/reaction.pyx, ILT branch)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _convolve(rho1: np.ndarray, rho2: np.ndarray) -> np.ndarray:
        """Discrete convolution (rmgpy/statmech/schrodinger.pyx convolve)."""
        if rho1.shape[0] != rho2.shape[0]:
            raise PDepNetworkError("Cannot convolve arrays of different length.")
        n = rho1.shape[0]
        rho = np.zeros(n, float)
        for i in range(n):
            for j in range(i + 1):
                rho[i] += rho2[i - j] * rho1[j]
        return rho

    def apply_ilt_k_e(self, e_list: np.ndarray, dens_states: np.ndarray,
                      A: float, n: float, Ea: float, E0_ts: float,
                      T: float = 0.0) -> np.ndarray:
        """
        The inverse-Laplace-transform microcanonical rate k(E) for a path
        reaction with a high-pressure-limit Arrhenius rate A*T^n*exp(-Ea/RT)
        and a TS ground-state energy E0_ts (no vibrational data -> the no-QM
        path). Ported from rmgpy/pdep/reaction.pyx
        apply_inverse_laplace_transform_method. Returns k(E) shaped like
        dens_states (n_grains, n_j).
        """
        n_grains = e_list.shape[0]
        n_j = dens_states.shape[1] if dens_states.ndim == 2 else 1
        k = np.zeros((n_grains, n_j), float)
        d_e = e_list[1] - e_list[0]
        if n < 0.25:
            # fold T^n into the pre-exponential (RMG's n_crit approximation)
            if T > 0.0:
                A = A * (T ** n)
            n = 0.0
        if Ea < 0.0:
            A = A * np.exp(-Ea / R / T)
            Ea = 0.0
        if n < 0.25:
            m0, rem = divmod(Ea, d_e)
            m = int(m0)
            if rem == 0:
                for s in range(n_j):
                    for r in range(m, n_grains):
                        if e_list[r] > E0_ts and dens_states[r, s] != 0:
                            k[r, s] = A * dens_states[r - m, s] / dens_states[r, s]
            else:
                for s in range(n_j):
                    for r in range(m + 1, n_grains):
                        if (e_list[r] > E0_ts and dens_states[r, s] != 0
                                and abs(dens_states[r - m, s]) > 1e-12
                                and abs(dens_states[r - m - 1, s]) > 1e-12):
                            num = dens_states[r - m, s] * (
                                dens_states[r - m - 1, s] / dens_states[r - m, s]) ** (
                                -rem / (e_list[r - m - 1] - e_list[r - m]))
                            k[r, s] = A * num / dens_states[r, s]
        else:
            from scipy.special import gamma
            phi0 = np.zeros(n_grains, float)
            for r in range(n_grains):
                energy = e_list[r] - e_list[0] - Ea
                if energy > 1:
                    phi0[r] = (energy / R) ** (n - 1.0)
            phi0 = phi0 * (d_e / R) / gamma(n)
            for s in range(n_j):
                phi = self._convolve(phi0, dens_states[:, s])
                for r in range(n_grains):
                    if dens_states[r, s] != 0:
                        k[r, s] = A * phi[r] / dens_states[r, s]
        return k

    # ------------------------------------------------------------------ #
    # Convenience driver: k(T,P) on a (T,P) grid (mirrors RMG's API)
    # ------------------------------------------------------------------ #
    def calculate_rate_coefficients(
        self,
        Tlist: Sequence[float],
        Plist: Sequence[float],
        method: str = "chemically-significant eigenvalues",
        state_at: "dict" = None,
    ) -> np.ndarray:
        """
        Compute K (n_T, n_P, n_cfg, n_cfg). `state_at` maps T -> the seeded
        per-T state (e_list, dens_isomer, dens_product, P_coll, Kij, Gnj, Fim,
        eq_ratios, coll_freqs -- one entry per P in Plist). Ported from
        rmgpy/pdep/network.py calculate_rate_coefficients (method dispatch on
        the long strings). The CSE (Allen) method is the step-04 deliverable;
        the others raise NotImplementedError (job-08, step-05/07).
        """
        method_l = method.lower()
        if method_l not in (
            "chemically-significant eigenvalues",
            "chemically-significant eigenvalues georgievskii",
            "modified strong collision",
            "reservoir state",
            "simulation least squares",
            "simulation least squares ode",
            "simulation least squares matrix exponential",
            "simulation least squares eigen",
        ):
            raise PDepNetworkError('Unknown method "%s".' % method)

        n_cfg = self.n_cfg
        K = np.zeros((len(Tlist), len(Plist), n_cfg, n_cfg), float)
        for t, T in enumerate(Tlist):
            for p, P in enumerate(Plist):
                if state_at is not None:
                    st = state_at[T]
                    self._apply_state(st, T, P, coll_freq=st["coll_freqs"][p])
                if method_l == "chemically-significant eigenvalues":
                    self.apply_cse_allen()
                elif method_l == "chemically-significant eigenvalues georgievskii":
                    raise NotImplementedError(
                        "georgievskii CSE is a job-08 deliverable (step-05/07).")
                else:
                    raise NotImplementedError(
                        "method %r is a job-08 deliverable (step-05/07)." % method)
                K[t, p] = self.K
        return K

    def _apply_state(self, st: dict, T: float, P: float, coll_freq: float) -> None:
        """Load a seeded per-T state + the coll_freq for the current P."""
        self.T = float(T)
        self.P = float(P)
        self.e_list = np.asarray(st["e_list"], dtype=float)
        self.j_list = np.asarray(st["j_list"], dtype=int)
        n_cfg = self.n_cfg
        n_grains = self.e_list.shape[0]
        n_j = self.j_list.shape[0]
        dens = np.zeros((n_cfg, n_grains, n_j), float)
        dens[0] = np.asarray(st["dens_isomer"], dtype=float)
        if st.get("dens_product") is not None:
            dens[self.n_isom + self.n_reac] = np.asarray(st["dens_product"], dtype=float)
        self.dens_states = dens
        self.eq_ratios = np.asarray(st["eq_ratios"], dtype=float)
        self.Kij = np.asarray(st["Kij"], dtype=float)
        self.Gnj = np.asarray(st["Gnj"], dtype=float)
        self.Fim = np.asarray(st["Fim"], dtype=float)
        # Collision: Mcoll(T,P) = coll_freq(T,P) * P_coll(T). P_coll is
        # P-independent (seeded from the RMG-Py reference); coll_freq ~ P.
        P_coll = np.asarray(st["P_coll"], dtype=float)
        n_isom = self.n_isom
        Mcoll = np.zeros((n_isom, n_grains, n_j, n_grains, n_j), float)
        Mcoll[0] = float(coll_freq) * P_coll
        self.coll_freq = np.array([float(coll_freq)], float)
        self.Mcoll = Mcoll
