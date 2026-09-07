"""
tests/test_collision_cse.py

job-07/step-05: the collision side of the master equation.

Checks (per the step file):
  1. Collision frequency: LJ params (transport DB style) + bath -> the
     frequency vs RMG-Py. Two non-circular references:
       (a) the single-bath (N2) cases recorded in step-04's baseline
           (gates/baselines/job07/toy_lindemann_ref.json "coll_freqs");
       (b) the single-bath AND multi-bath (N2 0.5 / Ar 0.5) cases in
           gates/baselines/job07/collision_ref.json (recorded by
           scripts/record_job07_step05_reference.py, run in rmg_env).
     (b) exercises RMG's LJ bath-averaging (sigma linear, epsilon
     geometric, mass linear) beyond the single-bath identity.
  2. The grain configuration / collision matrix: rmgpu's
     SingleExponentialDown.generate_collision_matrix (the grain -> grain
     collision mapping, P_coll) matches RMG-Py's recorded P_coll from both
     baselines (machine precision).
  3. The CSE k(T,P) full pipeline: compose rmgpu's OWN collision frequency
     and collision matrix into Mcoll (no RMG-Py collision state at all),
     seed the step-04 Network with it, and reproduce RMG-Py's recorded CSE
     (Allen) k(T,P) K_ref from toy_lindemann_ref.json. This is the pre-gate
     proof that the collision model + the network compose correctly.
  4. Collision efficiency: rmgpu's
     SingleExponentialDown.calculate_collision_efficiency matches RMG-Py's
     recorded Chang-Bozzelli-Dean values (collision_ref.json) on the toy
     network's real e_list / dens_states, across a spread of barriers.
  5. The missing-LJ fallback (estimate_lj_params): the fixed-LJ table
     (RMG get_transport_properties_via_lennard_jones_parameters) returns the
     documented constants per heavy-atom count, so a species with no
     transport entry degrades to a constant instead of crashing the ME.

Non-circular: every reference value is recorded by an independent RMG-Py run
(never rmgpu); rmgpu re-derives its own computation and compares.
"""
import os
import sys
import json

import numpy as np
import pytest

from rmgpu.pdep.network import Network
from rmgpu.pdep.collision import (
    CollisionError,
    LennardJones,
    SingleExponentialDown,
    build_collision_inputs,
    calculate_collision_frequency,
    estimate_lj_params,
    lj_from_transport_entry,
    molecular_weight_si,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOY_REF = os.path.join(REPO, "gates", "baselines", "job07", "toy_lindemann_ref.json")
COLL_REF = os.path.join(REPO, "gates", "baselines", "job07", "collision_ref.json")

# The toy isomer's LJ params + molar mass (identical to the reference
# recording scripts: scripts/record_job07_step04_reference.py build_isomer).
SPECIES_LJ = LennardJones.from_angstrom_K(5.94, 559.0)
SPECIES_MW_G_MOL = 74.07


@pytest.fixture(scope="module")
def toy_ref():
    assert os.path.exists(TOY_REF), (
        "toy_lindemann_ref.json missing - regenerate with:\n"
        "  /home/jackson/miniforge3/envs/rmg_env/bin/python "
        "scripts/record_job07_step04_reference.py")
    with open(TOY_REF) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def coll_ref():
    assert os.path.exists(COLL_REF), (
        "collision_ref.json missing - regenerate with:\n"
        "  /home/jackson/miniforge3/envs/rmg_env/bin/python "
        "scripts/record_job07_step05_reference.py")
    with open(COLL_REF) as f:
        return json.load(f)


def _toy_network(ref):
    """Build the step-04 toy Lindemann Network (same as test_pdep_network)."""
    nb = ref["network"]
    gp = ref["grain_params"]
    return Network(
        label="toy-lindemann",
        n_isom=nb["n_isom"], n_reac=nb["n_reac"], n_prod=nb["n_prod"],
        E0=np.array([nb["E0_isomer"], nb["E0_product"]]),
        E0_ts=np.array([nb["E0_ts"]]),
        Tmin=gp["Tmin"], Tmax=gp["Tmax"], Pmin=gp["Pmin"], Pmax=gp["Pmax"],
        grain_size=gp["max_grain_size"], grain_count=gp["min_grain_count"],
    )


# ------------------------------------------------------------------ #
# 1a. Collision frequency vs RMG-Py (single bath N2, step-04 baseline)
# ------------------------------------------------------------------ #
def test_collision_frequency_single_bath(toy_ref):
    """rmgpu's LJ collision frequency == RMG-Py's recorded coll_freqs."""
    # The toy isomer (E0=-1e5 J/mol, E0_ts=+3e4): the recorded state is keyed
    # by T; the per-(T,P) coll_freqs are the ground truth.
    for t_key, blk in toy_ref["snapshots"].items():
        T = float(t_key)
        for p, P in enumerate(toy_ref["kTp"]["Plist"]):
            f_ref = blk["coll_freqs"][p]
            f = calculate_collision_frequency(
                T, P, SPECIES_LJ, molecular_weight_si(SPECIES_MW_G_MOL),
                {LennardJones.from_angstrom_K(3.41, 124.0): 1.0},
                {LennardJones.from_angstrom_K(3.41, 124.0): molecular_weight_si(28.04)})
            assert f == pytest.approx(f_ref, rel=1e-9), (
                "T=%g P=%g: rmgpu %.8e vs RMG %.8e" % (T, P, f, f_ref))


# ------------------------------------------------------------------ #
# 1b. Collision frequency vs RMG-Py (single + multi bath, step-05 baseline)
# ------------------------------------------------------------------ #
def test_collision_frequency_multi_bath(coll_ref):
    """The LJ bath-averaging (sigma linear, eps geometric, mass linear) matches
    RMG-Py for a 0.5/0.5 N2/Ar bath (beyond the single-bath identity)."""
    cf = coll_ref["collision_frequency"]
    bath = cf["bath"]
    sp = cf["species"]
    sp_lj = LennardJones.from_angstrom_K(sp["sigma_angstrom"], sp["epsilon_K"])
    sp_mass = molecular_weight_si(sp["mw_g_mol"])
    lj = {LennardJones.from_angstrom_K(b["sigma_angstrom"], b["epsilon_K"])
          for b in bath.values()}
    mass = {LennardJones.from_angstrom_K(b["sigma_angstrom"], b["epsilon_K"]):
            molecular_weight_si(b["mw_g_mol"]) for b in bath.values()}
    n2_lj, ar_lj = None, None
    for name, b in bath.items():
        l = LennardJones.from_angstrom_K(b["sigma_angstrom"], b["epsilon_K"])
        if name == "N2":
            n2_lj = l
        elif name == "Ar":
            ar_lj = l

    # Single bath (N2)
    for rec in cf["single_bath_N2"]:
        f = calculate_collision_frequency(rec["T"], rec["P"], sp_lj, sp_mass,
                                          {n2_lj: 1.0}, {n2_lj: mass[n2_lj]})
        assert f == pytest.approx(rec["value"], rel=1e-9), (
            "single-bath T=%g P=%g: rmgpu %.8e vs RMG %.8e"
            % (rec["T"], rec["P"], f, rec["value"]))

    # Multi bath (N2 0.5 / Ar 0.5)
    for rec in cf["multi_bath_N2Ar"]:
        bath_mix = {n2_lj: 0.5, ar_lj: 0.5}
        f = calculate_collision_frequency(rec["T"], rec["P"], sp_lj, sp_mass,
                                          bath_mix, mass)
        assert f == pytest.approx(rec["value"], rel=1e-9), (
            "multi-bath T=%g P=%g: rmgpu %.8e vs RMG %.8e"
            % (rec["T"], rec["P"], f, rec["value"]))


# ------------------------------------------------------------------ #
# 2. The grain -> grain collision matrix (P_coll) vs RMG-Py
# ------------------------------------------------------------------ #
def _etm_from_ref(ref):
    """The toy isomer's SingleExponentialDown (alpha0 = 447.5*0.011962 kJ/mol)."""
    # step-04/05 recording scripts: alpha0 = 447.5 * 0.011962 kJ/mol = 5.353 kJ/mol
    # -> J/mol via the Energy branch (kJ/mol -> *1000).
    alpha0 = 447.5 * 0.011962 * 1000.0  # J/mol
    return SingleExponentialDown(alpha0=alpha0, T0=300.0, n=0.85)


def test_collision_matrix_grain_mapping(toy_ref):
    """rmgpu's generate_collision_matrix == RMG-Py's recorded P_coll
    (the grain -> grain collision mapping), at machine precision, for every
    recorded (T) state in the step-04 baseline, plus the
    build_collision_inputs factorization Mcoll = coll_freq * P_coll."""
    etm = _etm_from_ref(toy_ref)

    # (a) P_coll per T, coll_freqs per (T,P): the grain mapping itself.
    for t_key, blk in toy_ref["snapshots"].items():
        T = float(t_key)
        e = np.array(blk["e_list"])
        j = np.array(blk["j_list"])
        dens = np.array(blk["dens_isomer"])
        P_ref = np.array(blk["P_coll"])
        P = etm.generate_collision_matrix(T, dens, e, j)
        assert P.shape == P_ref.shape, \
            "T=%g shape %s vs %s" % (T, P.shape, P_ref.shape)
        rel = np.abs(P - P_ref) / np.maximum(np.abs(P_ref), 1e-300)
        rel = rel[np.isfinite(rel)]
        assert len(rel) == 0 or rel.max() < 1e-12, \
            "T=%g P_coll max rel diff %.3e >= 1e-12" % (T, rel.max() if len(rel) else 0.0)

    # (b) build_collision_inputs factorization: Mcoll = coll_freq * P_coll,
    #     using rmgpu's own coll_freq and P_coll (checked vs the stored values).
    blk = list(toy_ref["snapshots"].values())[0]
    T = float(list(toy_ref["snapshots"].keys())[0])
    e = np.array(blk["e_list"])
    j = np.array(blk["j_list"])
    dens = np.array(blk["dens_isomer"])
    cf_ref = blk["coll_freqs"][0]
    P_ref = np.array(blk["P_coll"])
    cf, P_coll, Mcoll = build_collision_inputs(
        T, toy_ref["kTp"]["Plist"][0], etm, dens, e, j,
        SPECIES_LJ, molecular_weight_si(SPECIES_MW_G_MOL),
        {LennardJones.from_angstrom_K(3.41, 124.0): 1.0},
        {LennardJones.from_angstrom_K(3.41, 124.0): molecular_weight_si(28.04)})
    assert cf == pytest.approx(cf_ref, rel=1e-9)
    assert np.allclose(Mcoll, cf_ref * P_ref, rtol=1e-12, atol=0.0)


# ------------------------------------------------------------------ #
# 3. The CSE k(T,P) full pipeline (the pre-gate proof)
# ------------------------------------------------------------------ #
def test_cse_ktp_full_pipeline(toy_ref):
    """Compose rmgpu's OWN collision frequency + collision matrix into Mcoll,
    seed the step-04 Network (no RMG-Py collision state at all), and
    reproduce RMG-Py's recorded CSE (Allen) k(T,P). This is the full-pipeline
    proof that the collision model + network compose correctly.

    rmgpu's P_coll is used for the (T,P)-independent grain mapping; the
    network's _apply_state recomputes Mcoll = coll_freq * P_coll exactly as
    RMG-Py does, but here coll_freq and P_coll are rmgpu's own.
    """
    nb = toy_ref["network"]
    net = _toy_network(toy_ref)
    etm = _etm_from_ref(toy_ref)
    Tlist = toy_ref["kTp"]["Tlist"]
    Plist = toy_ref["kTp"]["Plist"]
    K_ref = np.array(toy_ref["kTp"]["K_ref"])
    fwd_row, fwd_col = toy_ref["kTp"]["fwd_idx"]
    snapshots = {float(k): v for k, v in toy_ref["snapshots"].items()}

    # Build a per-T state where P_coll is RMG's recorded grain mapping (the
    # reference for the *configuration*), but the coll_freq fed to the
    # network is RMG's (the frequency is checked independently in test 1; the
    # pipeline test confirms the COMPOSITION). To make the pipeline test
    # exercise rmgpu's OWN collision model, we recompute P_coll via rmgpu and
    # feed rmgpu's coll_freq.
    K = np.zeros((len(Tlist), len(Plist), nb["n_isom"] + nb["n_reac"] + nb["n_prod"],
                  nb["n_isom"] + nb["n_reac"] + nb["n_prod"]), float)
    n_cfg = nb["n_isom"] + nb["n_reac"] + nb["n_prod"]
    for t, T in enumerate(Tlist):
        st = snapshots[float(T)]
        e = np.array(st["e_list"])
        j = np.array(st["j_list"])
        dens = np.array(st["dens_isomer"])
        for p, P in enumerate(Plist):
            # rmgpu's own collision inputs.
            cf, P_coll, Mcoll = build_collision_inputs(
                T, P, etm, dens, e, j,
                SPECIES_LJ, molecular_weight_si(SPECIES_MW_G_MOL),
                {LennardJones.from_angstrom_K(3.41, 124.0): 1.0},
                {LennardJones.from_angstrom_K(3.41, 124.0): molecular_weight_si(28.04)})
            # Seed the network with rmgpu's collision state, but RMG's
            # DoS / k(E) / eq_ratios (the step-04 seeded state).
            st_local = dict(st)
            st_local["P_coll"] = P_coll.tolist()  # rmgpu's grain mapping
            # _apply_state uses st["coll_freqs"][p] as the coll_freq.
            st_local["coll_freqs"] = [cf]
            net._apply_state(st_local, T, P, coll_freq=cf)
            net.apply_cse_allen()
            K[t, p] = net.K

    # Compare to RMG-Py's recorded K_ref (non-circular).
    worst = 0.0
    for t in range(len(Tlist)):
        for p in range(len(Plist)):
            d = np.abs(K[t, p] - K_ref[t, p])
            rel = d / np.maximum(np.abs(K_ref[t, p]), 1e-300)
            rel = rel[np.isfinite(rel)]
            if len(rel):
                worst = max(worst, float(np.max(rel)))
    assert worst < 1.0e-6, "full-pipeline CSE k(T,P) max rel diff %.3e >= 1e-6" % worst
    # Forward channel (isomer -> product) specifically.
    for t in range(len(Tlist)):
        for p in range(len(Plist)):
            kr = K_ref[t, p, fwd_row, fwd_col]
            kf = K[t, p, fwd_row, fwd_col]
            if kr > 0:
                assert kf == pytest.approx(kr, rel=1e-6), (
                    "fwd k(T=%g,P=%.3e) mismatch: %.4e vs %.4e" % (Tlist[t], Plist[p], kf, kr))


# ------------------------------------------------------------------ #
# 4. Collision efficiency (MSC Chang-Bozzelli-Dean factor)
# ------------------------------------------------------------------ #
def test_collision_efficiency(coll_ref):
    """rmgpu's calculate_collision_efficiency == RMG-Py's recorded values on
    the toy network's real e_list / dens_states, across a spread of barriers
    (spanning the (0,1) range)."""
    etm = SingleExponentialDown(
        alpha0=coll_ref["etm"]["alpha0_J_mol"], T0=coll_ref["etm"]["T0"],
        n=coll_ref["etm"]["n"])
    n_check = 0
    for blk in coll_ref["collision_efficiency"]:
        T = blk["T"]
        e = np.array(blk["e_list"])
        j = np.array(blk["j_list"])
        dens = np.array(blk["dens"])
        E0 = blk["E0_isomer"]
        for key, beta_ref in blk["eff"].items():
            offset = float(key.split("_")[-1])
            e_reac = E0 + offset
            beta = etm.calculate_collision_efficiency(T, e, j, dens, E0, e_reac)
            assert beta == pytest.approx(beta_ref, rel=1e-8, abs=1e-8), (
                "T=%g e_reac+%.0f: rmgpu %.6f vs RMG %.6f" % (T, offset, beta, beta_ref))
            n_check += 1
    assert n_check > 0


# ------------------------------------------------------------------ #
# 5. The missing-LJ fallback (fixed-LJ table)
# ------------------------------------------------------------------ #
def test_lj_fallback_table():
    """estimate_lj_params returns RMG-Py's documented fixed-LJ constants per
    heavy-atom count (the last-resort fallback so a missing LJ entry never
    crashes the ME). Values from
    rmgpy/data/transport.py get_transport_properties_via_lennard_jones_parameters."""
    # (n_heavy, sigma_m, epsilon_K)
    expected = {
        1: (3.758e-10, 148.6),
        2: (4.443e-10, 110.7),
        3: (5.118e-10, 237.1),
        4: (4.687e-10, 531.4),
        5: (5.784e-10, 341.1),
        6: (5.949e-10, 399.3),
        12: (5.949e-10, 399.3),  # >= 6 uses the last row
    }
    for n_heavy, (sigma_m, eps_K) in expected.items():
        lj = estimate_lj_params(n_heavy)
        assert lj.sigma == pytest.approx(sigma_m, rel=1e-12)
        assert lj.epsilon == pytest.approx(eps_K * 8.314472, rel=1e-12)


def test_lj_from_transport_entry():
    """lj_from_transport_entry maps a job-02 TransportEntry (sigma angstrom,
    epsilon K) to SI correctly."""
    class _Entry:
        sigma = 5.94
        epsilon = 559.0
    lj = lj_from_transport_entry(_Entry())
    assert lj.sigma == pytest.approx(5.94e-10, rel=1e-12)
    assert lj.epsilon == pytest.approx(559.0 * 8.314472, rel=1e-12)


def test_molecular_weight_si():
    """molecular_weight_si = (g/mol) * amu (RMG's per-molecule mass convention)."""
    assert molecular_weight_si(74.07) == pytest.approx(74.07 * 1.660538921e-27, rel=1e-12)
    assert molecular_weight_si(28.04) == pytest.approx(28.04 * 1.660538921e-27, rel=1e-12)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
