"""
rmgpu.pdep.collision
====================

job-07/step-05: the collision side of the pressure-dependent master equation.

Ports from RMG-Py (rmgpy/pdep/):
  - collision.pyx       SingleExponentialDown (the collision-energy-transfer
                        model) -> SingleExponentialDown (this module), incl.
                        generate_collision_matrix (the grain -> grain collision
                        transfer matrix P, P-I) and calculate_collision_efficiency
                        (the MSC Chang-Bozzelli-Dean factor, for job-08).
  - configuration.pyx   calculate_collision_frequency (the Lennard-Jones
                        collision frequency omega22, LJ params from the
                        transport DB + bath gas composition) ->
                        calculate_collision_frequency (this module).
  - cse.pyx             the CSE (Allen) k(T,P) extraction already lives on the
                        Network (step-04, network.py apply_cse_allen); this
                        module supplies the collision inputs it consumes. The
                        georgievskii variant (cse.pyx get_rate_coefficients_CSE_Advanced)
                        is a job-08 deliverable (it needs the Network's
                        isomer/reactant channel objects, not just the
                        state-seeded arrays) and is deliberately NOT ported here.

SI units throughout (J, K, Pa, mol). No QM, no Cython.

How it plugs into the step-04 Network
-------------------------------------
The Network is state-seeded: it consumes, per (T,P), the collision frequency
`coll_freq` and the collision matrix `Mcoll`. RMG-Py (network.py
calculate_collision_model) factors these exactly as

    Mcoll(T,P) = coll_freq(T,P) * P_coll(T)

where P_coll = energy_transfer_model.generate_collision_matrix(T, dens, e, j)
is P-independent (depends on T only through alpha = alpha0*(T/T0)^n) and
coll_freq ~ P. This module computes BOTH factors independently:

  - calculate_collision_frequency(T, P, species_lj, species_mw, bath_lj, bath_mw)
  - SingleExponentialDown.generate_collision_matrix(T, dens_states, e_list, j_list)

...and the test composes them into Mcoll and drives the step-04 Network to
reproduce RMG-Py's recorded CSE k(T,P) (the full-pipeline, non-circular proof
that the collision model + network compose correctly before the gate).

LJ parameter source + the missing-LJ fallback (a job-08/10 blocker if it
crashes): the LJ sigma/epsilon normally come from the job-02 TransportDB
(TransportEntry: sigma in angstrom, epsilon in K). If a species has NO
transport entry, RMG-Py estimates them (rmgpy/data/transport.py): first via
group additivity (needs Tc/Pc, which rmgpu has no source for), then, as the
last resort, via a FIXED Lennard-Jones table by heavy-atom count
(get_transport_properties_via_lennard_jones_parameters). This module ports
that last-resort table (estimate_lj_params) so a missing LJ entry degrades
to a documented constant instead of crashing the ME.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

# Match RMG-Py (rmgpy/constants.py) EXACTLY. A mismatch here breaks the
# k(T,P) parity (the collision frequency feeds the ME matrix directly).
# Re-used from network.py so the whole pdep package shares one constant set.
from rmgpu.pdep.network import R, Na, kB, PDepNetworkError


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class CollisionError(PDepNetworkError):
    """A collision-model construction / evaluation error (RMG CollisionError)."""


# --------------------------------------------------------------------------- #
# Lennard-Jones parameters + the missing-LJ fallback
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LennardJones:
    """Lennard-Jones collision parameters. SI: sigma in m, epsilon in J/mol.
    Frozen (immutable + hashable) so it can key the bath mappings."""
    sigma: float    # collision diameter, m
    epsilon: float  # well depth, J/mol

    @classmethod
    def from_angstrom_K(cls, sigma_angstrom: float, epsilon_K: float) -> "LennardJones":
        return cls(sigma=float(sigma_angstrom) * 1.0e-10,
                   epsilon=float(epsilon_K) * R)


def lj_from_transport_entry(entry) -> LennardJones:
    """
    Build a LennardJones from a job-02 TransportDB TransportEntry
    (sigma in angstrom, epsilon in K). The TransportDB stores the LJ params
    the transport DB actually holds (sigma/epsilon are the only fields
    calculate_collision_frequency consumes).
    """
    return LennardJones.from_angstrom_K(
        float(getattr(entry, "sigma", 0.0) or 0.0),
        float(getattr(entry, "epsilon", 0.0) or 0.0),
    )


# RMG-Py's last-resort fixed LJ table (rmgpy/data/transport.py
# get_transport_properties_via_lennard_jones_parameters), keyed by the number
# of heavy (non-hydrogen) atoms. sigma in m, epsilon in K. Used when a species
# has no transport-library entry AND no group-additivity critical-point data.
_LJ_FALLBACK_BY_HEAVY_ATOMS = (
    # (n_heavy, sigma_m, epsilon_K)
    (1, 3.758e-10, 148.6),
    (2, 4.443e-10, 110.7),
    (3, 5.118e-10, 237.1),
    (4, 4.687e-10, 531.4),
    (5, 5.784e-10, 341.1),
    (6, 5.949e-10, 399.3),  # RMG: count >= 6 uses this row
)


def estimate_lj_params(n_heavy: int) -> LennardJones:
    """
    RMG-Py's fixed-LJ last-resort estimate (see _LJ_FALLBACK_BY_HEAVY_ATOMS).
    `n_heavy` is the number of non-hydrogen atoms. count >= 6 -> the last row.
    This is the documented fallback so a species with no transport entry
    never crashes the master equation.
    """
    n_heavy = int(n_heavy)
    if n_heavy <= 1:
        sigma_m, eps_K = _LJ_FALLBACK_BY_HEAVY_ATOMS[0][1:]
    elif n_heavy == 2:
        sigma_m, eps_K = _LJ_FALLBACK_BY_HEAVY_ATOMS[1][1:]
    elif n_heavy == 3:
        sigma_m, eps_K = _LJ_FALLBACK_BY_HEAVY_ATOMS[2][1:]
    elif n_heavy == 4:
        sigma_m, eps_K = _LJ_FALLBACK_BY_HEAVY_ATOMS[3][1:]
    elif n_heavy == 5:
        sigma_m, eps_K = _LJ_FALLBACK_BY_HEAVY_ATOMS[4][1:]
    else:
        sigma_m, eps_K = _LJ_FALLBACK_BY_HEAVY_ATOMS[5][1:]
    return LennardJones(sigma=sigma_m, epsilon=eps_K * R)


# --------------------------------------------------------------------------- #
# The Lennard-Jones collision frequency (RMG configuration.pyx
# Configuration.calculate_collision_frequency)
# --------------------------------------------------------------------------- #
def _reduced_mass(mass_a: float, mass_b: float) -> float:
    """
    The reduced mass mu in kg (per molecule). RMG-Py's
    configuration.pyx uses `spec.molecular_weight.value_si` (kg/molecule) for
    each species; the reduced mass of the pair is then 1/(1/m_a + 1/m_b).
    Callers must pass per-molecule masses (kg), i.e. (molar mass g/mol)
    converted via the atomic mass unit. See the note on `mass` in
    calculate_collision_frequency.
    """
    return 1.0 / (1.0 / float(mass_a) + 1.0 / float(mass_b))


def calculate_collision_frequency(
    T: float,
    P: float,
    species_lj: LennardJones,
    species_mass: float,
    bath: Mapping[LennardJones, float],
    bath_masses: Mapping[LennardJones, float],
) -> float:
    """
    The Lennard-Jones collision frequency (Hz) of one isomer colliding with a
    mixture of bath species, ported from RMG-Py
    rmgpy/pdep/configuration.pyx Configuration.calculate_collision_frequency.

    sigma = 0.5*(sigma_species + sum_i frac_i * sigma_i)
    epsilon = sqrt(epsilon_species * prod_i epsilon_i ** frac_i)
    mu = 1 / (1/m_species + 1/sum_i frac_i * m_i)
    omega22 = 1.16145*Tred^-0.14874 + 0.52487*exp(-0.77320*Tred)
              + 2.16178*exp(-2.43787*Tred),  with  Tred = R*T/epsilon
    omega = omega22 * sqrt(8*kB*T/(pi*mu)) * pi*sigma^2 * (P/(kB*T))

    Parameters
    ----------
    T : K
    P : Pa
    species_lj : LennardJones
        LJ params of the colliding isomer (SI).
    species_mass : kg
        Per-molecule mass of the isomer (kg) = (molar mass g/mol) * amu, with
        amu = 1.660538921e-27 (RMG's constants.amu). Must be consistent with
        the bath masses below.
    bath : mapping LennardJones -> mole fraction
        The bath-gas composition. Keys are the bath species' LJ params; values
        are their mole fractions (summing to 1).
    bath_masses : mapping LennardJones -> per-molecule mass (kg)
        The per-molecule mass (kg) of each bath species, keyed by the same
        LennardJones object used in `bath`.

    Returns
    -------
    float : the collision frequency in s^-1 (Hz).
    """
    # Batch-average the bath LJ params + mass (RMG configuration.pyx loop).
    bath_sigma = 0.0
    bath_epsilon = 1.0
    bath_mw = 0.0
    for lj, frac in bath.items():
        bath_sigma += lj.sigma * frac
        bath_epsilon *= lj.epsilon ** frac
        bath_mw += bath_masses[lj] * frac
    if bath_mw <= 0.0:
        raise CollisionError(
            "Bath-gas mass must be positive (got %g kg); the reduced mass is "
            "undefined." % bath_mw)

    sigma = 0.5 * (species_lj.sigma + bath_sigma)
    epsilon = math.sqrt(species_lj.epsilon * bath_epsilon)
    mu = _reduced_mass(species_mass, bath_mw)
    gas_concentration = P / kB / T  # molecules/m^3 (RMG: P/(kB*T))

    # Neufeld-Janzen-Aspenburg collision integral omega22(T*).
    tred = R * T / epsilon
    omega22 = (1.16145 * tred ** (-0.14874)
               + 0.52487 * math.exp(-0.77320 * tred)
               + 2.16178 * math.exp(-2.43787 * tred))

    return omega22 * math.sqrt(8.0 * kB * T / math.pi / mu) * math.pi * sigma * sigma * gas_concentration


# --------------------------------------------------------------------------- #
# The collision-energy-transfer model (RMG collision.pyx SingleExponentialDown)
# --------------------------------------------------------------------------- #
class SingleExponentialDown:
    """
    A single-exponential-down collisional energy-transfer model
    (RMG collision.pyx SingleExponentialDown). alpha (the mean deactivating
    energy transferred per collision) scales with temperature as
    alpha = alpha0 * (T/T0)^n.

    All energies are J/mol, T in K. `alpha0` is the mean deactivating energy
    at the reference temperature T0 (J/mol); `n` is the temperature exponent
    (n=0 -> temperature-independent alpha).
    """

    def __init__(self, alpha0: float = 0.0, T0: float = 0.0, n: float = 0.0):
        self.alpha0 = float(alpha0)
        self.T0 = float(T0)
        self.n = float(n)

    def __repr__(self):
        return "SingleExponentialDown(alpha0={0!r}, T0={1!r}, n={2:g})".format(
            self.alpha0, self.T0, self.n)

    def get_alpha(self, T: float) -> float:
        """The mean deactivating energy <dE_down> at temperature T (J/mol)."""
        if self.T0 <= 0.0:
            return self.alpha0
        return self.alpha0 * (T / self.T0) ** self.n

    # ------------------------------------------------------------------ #
    # The grain -> grain collision transfer matrix (RMG
    # collision.pyx generate_collision_matrix). Returns P (the transition
    # probability matrix, = P_coll in the Network factorization), shape
    # (n_grains, n_j, n_grains, n_j).
    # ------------------------------------------------------------------ #
    def generate_collision_matrix(self, T: float, dens_states: np.ndarray,
                                  e_list: np.ndarray,
                                  j_list: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate and return the collisional energy-transfer probability matrix
        P (RMG collision.pyx generate_collision_matrix). `dens_states` is the
        isomer density of states (n_grains, n_j); `e_list` (J/mol), `T` (K),
        `j_list` (int, optional). Returns P of shape (n_grains, n_j, n_grains,
        n_j). RMG's Network then forms Mcoll = coll_freq * P (network.py
        calculate_collision_model); that multiplication happens OUTSIDE this
        method (the Network's seed_collision does it), matching RMG's layout.
        """
        dens_states = np.asarray(dens_states, dtype=float)
        e_list = np.asarray(e_list, dtype=float)
        n_grains = e_list.shape[0]
        if j_list is None:
            j_list = np.array([0], dtype=int)
        j_list = np.asarray(j_list, dtype=int)
        n_j = j_list.shape[0]

        p = np.zeros((n_grains, n_j, n_grains, n_j), float)
        p0 = np.zeros((n_grains, n_grains), float)

        alpha = 1.0 / self.get_alpha(T)
        beta = 1.0 / (R * T)

        # Reduced 1-D DoS (sum over J with (2J+1) degeneracy).
        if n_j > 1:
            rho = np.sum((2 * j_list + 1) * dens_states, axis=1)
        else:
            rho = dens_states[:, 0].copy()

        # First grain with nonzero DoS (RMG: `for start ... if rho>0: break`).
        nz = np.flatnonzero(rho > 0)
        start = int(nz[0]) if nz.size else 0

        # Determine unnormalized entries (ported EXACTLY, nested loops, from
        # RMG collision.pyx). p0[s, r] for a fixed column r:
        #   s in [start, r]:   p0[s,r] = exp(-(e_r - e_s) alpha)
        #   s in (r, n_grains): p0[s,r] = exp(-(e_s - e_r) alpha)
        #                          * rho_s/rho_r * exp(-(e_s - e_r) beta)
        # (n_grains ~ 100-500; consistent with the sequential normalization
        # loop below -- both are O(n^2) pure-Python and cheap.)
        for r in range(start, n_grains):
            for s in range(start, r + 1):
                p0[s, r] = math.exp(-(e_list[r] - e_list[s]) * alpha)
            for s in range(r + 1, n_grains):
                p0[s, r] = (math.exp(-(e_list[s] - e_list[r]) * alpha)
                            * rho[s] / rho[r]
                            * math.exp(-(e_list[s] - e_list[r]) * beta))

        # Detailed-balance normalization -- ported EXACTLY (sequentially) from
        # RMG collision.pyx. The loop is sequential BY DESIGN: each c_r is read
        # from p0 values already scaled by earlier c's, so it cannot be
        # vectorized without changing the result. (n_grains ~ 100-500 -> cheap.)
        for r in range(start, n_grains):
            left = 0.0
            right = 0.0
            for s in range(start, r):
                left += p0[s, r]
            for s in range(r, n_grains):
                right += p0[s, r]
            c = (1.0 - left) / right
            # Check for normalization consistency (i.e. all numbers positive).
            if c < 0:
                raise CollisionError(
                    "Encountered negative normalization coefficient while "
                    "normalizing collisional transfer probabilities matrix.")
            for s in range(r + 1, n_grains):
                p0[r, s] *= c
                p0[s, r] *= c
            p0[r, r] = p0[r, r] * c - 1.0

        # 2D (J-active): P(E,J;E',J') = P0(E;E') * phi(E,J), with the strong-
        # collision approximation in J (RMG collision.pyx).
        if n_j > 1:
            phi = np.zeros_like(dens_states)
            for s in range(n_j):
                phi[:, s] = (2 * j_list[s] + 1) * dens_states[:, s]
            for r in range(start, n_grains):
                if rho[r] > 0:
                    phi[r, :] /= rho[r]
            for s in range(n_j):
                for v in range(n_j):
                    p[:, s, :, v] = p0 * phi[:, s]
        else:
            p[:, 0, :, 0] = p0

        return p

    # ------------------------------------------------------------------ #
    # Collision efficiency (RMG collision.pyx calculate_collision_efficiency)
    # -- the Chang-Bozzelli-Dean factor used by the MSC method (job-08).
    # Ported now so job-08 only adds the MSC driver onto this model.
    # ------------------------------------------------------------------ #
    def calculate_collision_efficiency(self, T: float, e_list: np.ndarray,
                                       j_list: Optional[np.ndarray],
                                       dens_states: np.ndarray,
                                       E0: float, e_reac: float) -> float:
        """
        A collision-efficiency factor (0..1) for the MSC method, ported from
        RMG collision.pyx calculate_collision_efficiency (Chang, Bozzelli,
        Dean, Z. Phys. Chem. 214, 1533 (2000)). `E0` (J/mol) is the isomer
        ground state, `e_reac` (J/mol) the first reactive energy, `dens_states`
        (n_grains, n_j), `T` (K).
        """
        e_list = np.asarray(e_list, dtype=float)
        dens_states = np.asarray(dens_states, dtype=float)
        if j_list is None:
            j_list = np.array([0], dtype=int)
        j_list = np.asarray(j_list, dtype=int)
        n_grains = e_list.shape[0]
        n_j = j_list.shape[0]
        d_e = e_list[1] - e_list[0]
        beta = 1.0 / (R * T)

        if e_reac - E0 < 100.0:
            e_reac = E0 + 100.0
        d_e_down = self.get_alpha(T)

        # Boltzmann-weighted population per grain. The Boltzmann factor depends
        # only on the grain r (shape (n,)), so it must be expanded to (n,1) to
        # broadcast across the J axis -- NOT (n,), which would broadcast
        # (n,n_j)*(n,) -> (n,n) and corrupt the sum.
        boltz = np.exp(-e_list * beta)[:, None]
        weight = np.sum(dens_states * (2 * j_list + 1)[None, :] * boltz, axis=1)

        fe_num = 0.0
        fe_den = 0.0
        for r in range(n_grains):
            if e_list[r] > e_reac:
                fe_num += weight[r]
                if fe_den == 0.0:
                    fe_den = weight[r] * R * T / d_e
        if fe_den == 0.0:
            return 1.0
        fe = fe_num / fe_den
        if fe > 1.0e6:
            fe = 1.0e6  # RMG: freeze fe > 1e6 (roundoff), per Chang et al.

        delta1 = 0.0
        delta2 = 0.0
        delta_n = 0.0
        for r in range(n_grains):
            delta_n += weight[r]
            if e_list[r] < e_reac:
                delta1 += weight[r]
                delta2 += weight[r] * math.exp(-(e_reac - e_list[r]) / (fe * R * T))
        delta1 /= delta_n
        delta2 /= delta_n
        delta = delta1 - (fe * R * T) / (d_e_down + fe * R * T) * delta2
        beta_eff = (d_e_down / (d_e_down + fe * R * T)) ** 2 / delta
        if beta_eff > 1.0:
            beta_eff = 1.0
        if beta_eff < 0.0:
            raise CollisionError(
                "Invalid collision efficiency %g calculated at %g K." % (beta_eff, T))
        return beta_eff


# --------------------------------------------------------------------------- #
# Convenience: build the full collision input for a single isomer
# --------------------------------------------------------------------------- #
def build_collision_inputs(T: float, P: float, model: SingleExponentialDown,
                           dens_states: np.ndarray, e_list: np.ndarray,
                           j_list: Optional[np.ndarray],
                           species_lj: LennardJones, species_mass: float,
                           bath: Mapping[LennardJones, float],
                           bath_masses: Mapping[LennardJones, float]) -> tuple:
    """
    Build (coll_freq, P_coll, Mcoll) for a single isomer at (T, P):
      - coll_freq = calculate_collision_frequency(...)  (s^-1)
      - P_coll = model.generate_collision_matrix(T, dens, e, j)  (grain -> grain
        transfer probabilities, P-independent)
      - Mcoll = coll_freq * P_coll  (the Network's seed_collision input,
        matching RMG network.py calculate_collision_model).
    """
    coll_freq = calculate_collision_frequency(T, P, species_lj, species_mass,
                                              bath, bath_masses)
    P_coll = model.generate_collision_matrix(T, dens_states, e_list, j_list)
    Mcoll = coll_freq * P_coll
    return float(coll_freq), P_coll, Mcoll


# AMU constant (RMG rmgpy/constants.py amu = 1.660538921e-27 kg) for converting
# a molar mass in g/mol to a per-molecule mass in kg, consistent with RMG's
# molecular_weight.value_si convention used by calculate_collision_frequency.
AMU = 1.660538921e-27


def molecular_weight_si(molar_mass_g_mol: float) -> float:
    """Per-molecule mass (kg) = (molar mass g/mol) * amu (RMG convention)."""
    return float(molar_mass_g_mol) * AMU
