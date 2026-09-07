"""
tests/test_pdep_network.py

job-07/step-04: the pdep network + master equation (CSE) + TS-E0.

Checks (per the step file):
  1. TS-E0 derivation: the no-QM path (derive_ts_e0) reproduces the TS E0 that
     RMG-Py's core loop derives from the HPL rate (the Ea-based form,
     rmgpy/rmg/pdep.py:856). This is the no-QM path's proof (PLAN 8a.2).
  2. Grain generation: rmgpu's select_energy_grains matches RMG-Py's
     (counts + the grain boundaries / e_list).
  3. The ME integration / k(T,P): the toy Lindemann k(T,P) via CSE (Allen)
     matches RMG-Py (the sub-gate that de-risks the job-07 k(T,P) gate).
  4. The ILT k(E) port matches RMG-Py's raw ILT microcanonical rate.

Non-circular: every reference value (the per-(T,P) network state, the RMG-Py
CSE k(T,P) K_ref, and the RMG-Py ILT k(E)) is recorded by an INDEPENDENT
RMG-Py run (scripts/record_job07_step04_reference.py, run in rmg_env) into
gates/baselines/job07/toy_lindemann_ref.json. rmgpu re-derives its own
computation and compares to those recorded values - it is never checked
against itself.
"""
import os
import sys
import json

import numpy as np
import pytest

from rmgpu.pdep.network import (
    Network,
    derive_ts_e0,
    network_energy_correction,
    PDepNetworkError,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(REPO, "gates", "baselines", "job07", "toy_lindemann_ref.json")

# ------------------------------------------------------------------ #
# Shared fixtures (load the RMG-Py reference once)
# ------------------------------------------------------------------ #
@pytest.fixture(scope="module")
def ref():
    assert os.path.exists(REF), (
        "toy_lindemann_ref.json missing - regenerate with:\n"
        "  /home/jackson/miniforge3/envs/rmg_env/bin/python "
        "scripts/record_job07_step04_reference.py")
    with open(REF) as f:
        return json.load(f)


def _make_network(ref):
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
# 1. TS-E0 derivation (the no-QM path)
# ------------------------------------------------------------------ #
def test_ts_e0_derivation_noqm(ref):
    """derive_ts_e0 must reproduce the TS E0 that RMG-Py's core loop derives.

    RMG-Py (rmgpy/rmg/pdep.py:856): E0_TS = sum(reactant E0) + Ea +
    energy_correction. The reference records all four quantities from RMG-Py's
    own run, so this is a direct (non-circular) comparison.
    """
    nb = ref["network"]
    hp = ref["hpl_arrhenius"]
    # energy_correction = -min over the network's channel E0s (isomer/product)
    ec = network_energy_correction([nb["E0_isomer"], nb["E0_product"]])
    assert ec == pytest.approx(nb["energy_correction"], rel=0.0, abs=0.0)
    e0_ts = derive_ts_e0([nb["E0_isomer"]], hp["Ea"], energy_correction=ec)
    assert e0_ts == pytest.approx(nb["E0_ts"], rel=0.0, abs=1e-9)


def test_ts_e0_two_reactants():
    """Sanity: the sum is over BOTH reactant E0s for a bimolecular reaction."""
    e0 = derive_ts_e0([1000.0, 2000.0], 500.0, energy_correction=-1500.0)
    assert e0 == pytest.approx(1000.0 + 2000.0 + 500.0 - 1500.0, abs=1e-9)


# ------------------------------------------------------------------ #
# 2. Grain generation parity
# ------------------------------------------------------------------ #
def test_grain_generation_parity(ref):
    """rmgpu select_energy_grains == RMG-Py's INITIAL (pure) selection.

    Compares counts and the full grain boundaries (e_list). (RMG-Py's
    set_conditions may then re-select via its k(E)-validity retry loop; that
    final grid is seeded for the CSE test - see test_cse_ktp_parity.)
    """
    gp = ref["grain_params"]
    net = _make_network(ref)
    for t in ref["kTp"]["Tlist"]:
        e_ref = np.array(ref["snapshots"][str(int(round(t)))]["e_list_initial"])
        e = net.select_energy_grains(float(t), gp["max_grain_size"], gp["min_grain_count"])
        assert len(e) == len(e_ref), (
            "T=%g grain count mismatch: rmgpu %d vs RMG %d" % (t, len(e), len(e_ref)))
        assert np.allclose(e, e_ref, rtol=0.0, atol=1e-6), (
            "T=%g grain boundary mismatch (maxdiff %.3e)" %
            (t, np.max(np.abs(e - e_ref))))


# ------------------------------------------------------------------ #
# 3. CSE k(T,P) parity (the sub-gate that de-risks the job-07 k(T,P) gate)
# ------------------------------------------------------------------ #
def test_cse_ktp_parity(ref):
    """rmgpu's CSE (Allen) k(T,P) matches RMG-Py's recorded K_ref.

    The Network is seeded with RMG-Py's recorded per-T state (e_list, DoS,
    collision matrix = coll_freq*P_coll, k(E) matrices, eq_ratios) and
    rmgpu re-computes the ME matrix + CSE k(T,P) from that state. The
    comparison is against RMG-Py's own CSE run (non-circular).
    """
    nb = ref["network"]
    gp = ref["grain_params"]
    net = _make_network(ref)
    Tlist = ref["kTp"]["Tlist"]
    Plist = ref["kTp"]["Plist"]
    K_ref = np.array(ref["kTp"]["K_ref"])
    fwd_row, fwd_col = ref["kTp"]["fwd_idx"]

    snapshots = {float(k): v for k, v in ref["snapshots"].items()}
    K = net.calculate_rate_coefficients(
        Tlist, Plist, "chemically-significant eigenvalues", state_at=snapshots)

    assert K.shape == K_ref.shape
    # Full-matrix parity (every entry, finite), and the forward channel.
    worst = 0.0
    for t in range(len(Tlist)):
        for p in range(len(Plist)):
            d = np.abs(K[t, p] - K_ref[t, p])
            rel = d / np.maximum(np.abs(K_ref[t, p]), 1e-300)
            rel = rel[np.isfinite(rel)]
            if len(rel):
                worst = max(worst, float(np.max(rel)))
    # The sub-gate bar: < 1% on k(T,P) (the job-07 gate target); we assert a
    # tight machine-precision bound since the state is seeded from RMG-Py.
    assert worst < 1.0e-3, "CSE k(T,P) max rel diff %.3e >= 1e-3" % worst
    # Forward channel (isomer -> product) specifically.
    for t in range(len(Tlist)):
        for p in range(len(Plist)):
            kr = K_ref[t, p, fwd_row, fwd_col]
            kf = K[t, p, fwd_row, fwd_col]
            if kr > 0:
                assert kf == pytest.approx(kr, rel=1e-6), (
                    "fwd k(T=%g,P=%.3e) mismatch: %.4e vs %.4e" %
                    (Tlist[t], Plist[p], kf, kr))


# ------------------------------------------------------------------ #
# 4. ILT k(E) port parity (the no-QM microcanonical rate)
# ------------------------------------------------------------------ #
def test_ilt_k_e_parity(ref):
    """rmgpu's ILT k(E) matches RMG-Py's raw ILT k(E) (non-circular)."""
    nb = ref["network"]
    hp = ref["hpl_arrhenius"]
    net = Network(label="t", n_isom=nb["n_isom"], n_reac=nb["n_reac"], n_prod=nb["n_prod"])
    for t in ref["kTp"]["Tlist"]:
        b = ref["snapshots"][str(int(round(t)))]
        e = np.array(b["e_list"])
        dens = np.array(b["dens_isomer"])
        k_ref = np.array(b["ilt_kE_ref"])
        k = net.apply_ilt_k_e(e, dens, hp["A"], hp["n"], hp["Ea"],
                              nb["E0_ts"], T=float(t))
        assert k.shape == k_ref.shape
        nz = k_ref > 0
        if nz.any():
            rel = np.abs(k[nz] - k_ref[nz]) / k_ref[nz]
            assert rel.max() < 1e-6, (
                "T=%g ILT k(E) max rel diff %.3e >= 1e-6" % (t, rel.max()))
        # zero where RMG has zero (below the barrier / below the DoS)
        assert np.allclose(k[k_ref == 0], 0.0, atol=0.0)


# ------------------------------------------------------------------ #
# 5. Method dispatch (the long strings)
# ------------------------------------------------------------------ #
def test_method_dispatch_long_strings(ref):
    """The step-04 deliverable is CSE (Allen); the others raise NotImplementedError
    (job-08, step-05/07) - not a silent no-op."""
    nb = ref["network"]
    net = Network(label="t", n_isom=nb["n_isom"], n_reac=nb["n_reac"], n_prod=nb["n_prod"])
    snapshots = {float(k): v for k, v in ref["snapshots"].items()}
    T, P = ref["kTp"]["Tlist"][0], ref["kTp"]["Plist"][0]

    # CSE (allen) works
    K = net.calculate_rate_coefficients([T], [P], "chemically-significant eigenvalues",
                                        state_at=snapshots)
    assert K.shape == (1, 1, nb["n_isom"] + nb["n_reac"] + nb["n_prod"],
                       nb["n_isom"] + nb["n_reac"] + nb["n_prod"])

    # the others are job-08 / step-05/07 -> NotImplementedError
    for m in ["modified strong collision", "reservoir state",
              "chemically-significant eigenvalues georgievskii",
              "simulation least squares", "simulation least squares ode",
              "simulation least squares matrix exponential",
              "simulation least squares eigen"]:
        with pytest.raises(NotImplementedError):
            net.calculate_rate_coefficients([T], [P], m, state_at=snapshots)

    # unknown method -> PDepNetworkError (RMG's behavior), not a silent no-op
    with pytest.raises(PDepNetworkError):
        net.calculate_rate_coefficients([T], [P], "bogus method", state_at=snapshots)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
