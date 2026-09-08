"""job-07/step-06: the pdep DRIVER test (the wiring test).

Runs a hand-built (here: RMG-Py-recorded) 2-ISOMER pressure-dependent network
through ``rmgpu.pdep.driver.run_pdep`` and checks the three wiring invariants
from the step file:

  (a) the Falloff attaches - every net reaction's rate model is now a fitted
      Chebyshev / PDepArrhenius (the "Falloff"), not an HPL Arrhenius;
  (b) pdep/<network>.yaml writes + parses (the self-contained network
      definition + k(T,P) grid, PLAN 12.3);
  (c) the fit reproduces the (T,P) grid within the interpolation tolerance
      (the fit's error is within RMG's error_check bar of 0.5 in log-RMS, and
      the evaluated k(T,P) tracks the grid's k(T,P) point-for-point).

NON-CIRCULAR: the reference (gates/baselines/job07/toy_2isomer_ref.json) is
RMG-Py's OWN run of the 2-isomer network (recorded by
scripts/record_job07_step06_reference.py in rmg_env). It records the per-T
network STATE (DoS, fluxes Kij/Gnj/Fim, collision) so rmgpu's Network solves
the master equation from that state, plus RMG-Py's own CSE k(T,P) matrix
K_ref on the (T,P) grid. The driver's solve is compared against K_ref
(wiring: the network + state seam + net-reaction pairing are correct), and the
fit is compared against the driver's OWN grid (wiring: the fit + attach are
correct). The gate (step 07) does the CSE k(T,P) parity at full precision.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from rmgpu.pdep.network import Network
from rmgpu.pdep import driver
from rmgpu.kinetics.models import Chebyshev, PDepArrhenius

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_PATH = os.path.join(HERE, "gates", "baselines", "job07", "toy_2isomer_ref.json")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def ref():
    """The committed RMG-Py 2-isomer reference (recorded, non-circular)."""
    with open(REF_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def reference_data(ref):
    """(Tlist, Plist, K_ref, net_reactions, snapshot provider) - the RMG-Py
    recorded values the driver is checked against."""
    Tlist = np.array(ref["kTp"]["Tlist"], dtype=float)
    Plist = np.array(ref["kTp"]["Plist"], dtype=float)
    K_ref = np.array(ref["kTp"]["K_ref"], dtype=float)
    net_reactions = ref["net_reactions"]
    snapshots = ref["snapshots"]

    def provider(T):
        # The snapshot keys are str(round(T, 6)); the grid T is the exact
        # Gauss-Chebyshev value the recorder used, so this resolves.
        return snapshots[str(round(float(T), 6))]

    return Tlist, Plist, K_ref, net_reactions, provider


@pytest.fixture(scope="module")
def pdep_network(reference_data):
    """Build the rmgpu step-04 Network (2 isomers, 1 product channel) + the
    driver's PDepNetwork view with the recorded state provider and the RMG
    net-reaction (from,to) pairs. A 2-isomer topology - no numerics change,
    pure orchestration (step-04's CSE is general over n_isom)."""
    Tlist, Plist, K_ref, net_reactions, provider = reference_data
    nb = _network_shape()
    gp = _grain_params()
    net = Network(
        label="toy-2isomer",
        n_isom=nb["n_isom"], n_reac=nb["n_reac"], n_prod=nb["n_prod"],
        E0=np.array(nb["E0_isomers"] + [nb["E0_product"]]),
        Tmin=gp["Tmin"], Tmax=gp["Tmax"], Pmin=gp["Pmin"], Pmax=gp["Pmax"],
        grain_size=gp["max_grain_size"], grain_count=gp["min_grain_count"],
    )
    net.state_provider = provider
    pdn = driver.PDepNetwork(label="toy-2isomer", network=net)
    for nr in net_reactions:
        pdn.reactions.append(driver.PDepReaction(
            label="cfg%d>cfg%d" % (nr["from_cfg"], nr["to_cfg"]),
            from_cfg=nr["from_cfg"], to_cfg=nr["to_cfg"],
            n_reactants=nr["n_reactants"]))
    return pdn


def _network_shape():
    with open(REF_PATH) as f:
        return json.load(f)["network"]


def _grain_params():
    with open(REF_PATH) as f:
        return json.load(f)["grain_params"]


# --------------------------------------------------------------------------- #
# (a) the Falloff attaches
# --------------------------------------------------------------------------- #

def test_falloff_attaches_chebyshev(pdep_network, reference_data):
    Tlist, Plist, K_ref, _, _ = reference_data
    res = driver.run_pdep(pdep_network, method="cse",
                          interpolation_model=("Chebyshev", 6, 4),
                          Tlist=Tlist, Plist=Plist, error_check=True)
    assert res.n_fitted == len(pdep_network.reactions)
    for rxn in res.reactions:
        # The rate model is now a FITTED Chebyshev (the Falloff), not an HPL
        # Arrhenius. The (from,to) pairing is correct (the 4 net reactions of
        # the 2-isomer + 1-product network).
        assert isinstance(rxn.kinetics, Chebyshev), \
            "Falloff did not attach for %s (got %r)" % (rxn.label, type(rxn.kinetics))
        assert rxn.kinetics.degreeT == 6
        assert rxn.kinetics.degreeP == 4
        assert rxn.kinetics.coeffs.shape == (6, 4)
        # The fit is on the SI scale: evaluating gives SI k(T,P) (s^-1 here -
        # the isomer channels are unimolecular).
        k = rxn.kinetics.get_rate_coefficient(500.0, 1.0e5)
        assert np.isfinite(k)


def test_falloff_attaches_pdeparrhenius(pdep_network, reference_data):
    Tlist, Plist, K_ref, _, _ = reference_data
    res = driver.run_pdep(pdep_network, method="cse",
                          interpolation_model=("PDepArrhenius",),
                          Tlist=Tlist, Plist=Plist, error_check=True)
    for rxn in res.reactions:
        assert isinstance(rxn.kinetics, PDepArrhenius), \
            "Falloff did not attach for %s (got %r)" % (rxn.label, type(rxn.kinetics))
        assert rxn.kinetics.pressures is not None
        assert len(rxn.kinetics.pressures) == len(Plist)
        assert len(rxn.kinetics.arrhenius) == len(Plist)


def test_driver_selects_cse_only(pdep_network, reference_data):
    # Method selection: CSE works; MSC / RS / SLS raise NotImplementedError
    # (job 08); unknown methods raise PDepDriverError (RMG's
    # PressureDependenceError analogue).
    Tlist, Plist, K_ref, _, _ = reference_data
    with pytest.raises(NotImplementedError):
        driver.run_pdep(pdep_network, method="masc",
                        interpolation_model=("Chebyshev", 6, 4),
                        Tlist=Tlist, Plist=Plist)
    with pytest.raises(NotImplementedError):
        driver.run_pdep(pdep_network, method="rs",
                        interpolation_model=("Chebyshev", 6, 4),
                        Tlist=Tlist, Plist=Plist)
    with pytest.raises(NotImplementedError):
        driver.run_pdep(pdep_network, method="sls",
                        interpolation_model=("Chebyshev", 6, 4),
                        Tlist=Tlist, Plist=Plist)
    with pytest.raises(driver.PDepDriverError):
        driver.resolve_method("not-a-method")


# --------------------------------------------------------------------------- #
# (b) pdep/<network>.yaml writes + parses
# --------------------------------------------------------------------------- #

def test_network_yaml_writes_and_parses(pdep_network, reference_data, tmp_path):
    Tlist, Plist, K_ref, net_reactions, _ = reference_data
    out_dir = str(tmp_path)
    res = driver.run_pdep(pdep_network, method="cse",
                          interpolation_model=("Chebyshev", 6, 4),
                          Tlist=Tlist, Plist=Plist, output_dir=out_dir,
                          error_check=True)
    assert res.yaml_path is not None
    assert os.path.isfile(res.yaml_path)
    # pdep/<network>.yaml (the file lives under output_dir/pdep/).
    assert res.yaml_path.endswith("pdep/toy-2isomer.yaml")
    doc = driver.load_network_yaml(res.yaml_path)
    # The self-contained network definition (PLAN 12.3): structure + grid +
    # the fitted coefficients + the k(T,P) grid.
    assert doc["network"] == "toy-2isomer"
    assert doc["n_isom"] == 2
    assert doc["n_reac"] == 0
    assert doc["n_prod"] == 1
    assert doc["n_cfg"] == 3
    assert len(doc["Tlist"]) == len(Tlist)
    assert len(doc["Plist"]) == len(Plist)
    assert len(doc["reactions"]) == len(net_reactions)
    # The k(T,P) grid is recorded (self-contained).
    K_grid = np.array(doc["K_grid"])
    assert K_grid.shape == (len(Tlist), len(Plist), 3, 3)
    # Each reaction carries its fitted kinetics (the coefficients round-trip).
    for rxn_doc in doc["reactions"]:
        kin = rxn_doc["kinetics"]
        assert kin is not None
        assert kin["type"] == "Chebyshev"
        assert len(kin["coeffs"]) == 6
        assert len(kin["coeffs"][0]) == 4


# --------------------------------------------------------------------------- #
# (c) the fit reproduces the (T,P) grid within the interpolation tolerance
# --------------------------------------------------------------------------- #

def test_fit_reproduces_grid(pdep_network, reference_data):
    """The fit's evaluated k(T,P) tracks the grid's k(T,P): within the
    interpolation tolerance (RMG's error_check bar of 0.5 in log-RMS, and
    point-for-point within a fraction of an order of magnitude)."""
    Tlist, Plist, K_ref, net_reactions, _ = reference_data
    # Both interpolation models reproduce their own grid.
    for model in (("Chebyshev", 6, 4), ("PDepArrhenius",)):
        res = driver.run_pdep(pdep_network, method="cse",
                              interpolation_model=model,
                              Tlist=Tlist, Plist=Plist, error_check=True)
        # RMG's error_check: every net reaction's fit log-RMS < 0.5.
        for rxn in res.reactions:
            assert rxn.fit_log_rms < 0.5, \
                "fit log-RMS %.4f >= 0.5 for %s (model %r)" % (
                    rxn.fit_log_rms, rxn.label, model)
        # Point-for-point: the evaluated k(T,P) is within a fraction of an
        # order of magnitude of the grid's k(T,P) (the interpolation tolerance
        # - the fit is a low-degree surrogate for the grid).
        for rxn in res.reactions:
            if rxn.kinetics is None:
                continue
            kdata = res.K[:, :, rxn.to_cfg, rxn.from_cfg]
            worst = 0.0
            for t in range(len(Tlist)):
                for p in range(len(Plist)):
                    km = rxn.kinetics.get_rate_coefficient(Tlist[t], Plist[p])
                    kd = kdata[t, p]
                    if kd > 0.0 and km > 0.0:
                        worst = max(worst, abs(km - kd) / kd)
            assert worst < 1.0, \
                "fit deviates >100%% from the grid for %s (model %r)" % (
                    rxn.label, model)


def test_network_solve_matches_rmg_reference(pdep_network, reference_data):
    """The driver's network SOLVE (CSE k(T,P) from the recorded RMG-Py state)
    matches RMG-Py's OWN K_ref on the (T,P) grid - the wiring (network build +
    state seam + net-reaction pairing + grid) is correct. This is the
    non-circular cross-check; the gate (step 07) re-derives K_ref at full
    precision (the gate's k(T,P) parity)."""
    Tlist, Plist, K_ref, net_reactions, _ = reference_data
    res = driver.run_pdep(pdep_network, method="cse",
                          interpolation_model=("Chebyshev", 6, 4),
                          Tlist=Tlist, Plist=Plist, error_check=False)
    K = res.K
    # The driver reproduces RMG's k(T,P) matrix. Allow a small relative
    # tolerance (CSE eigen-decomposition is exact to machine precision, but the
    # per-T state is seeded from the recorded snapshot; the grid is the same).
    denom = np.maximum(np.abs(K_ref), 1.0e-300)
    rel = np.abs(K - K_ref) / denom
    rel = rel[np.isfinite(rel)]
    assert rel.max() < 1.0e-3, \
        "network solve k(T,P) deviates from RMG K_ref by %.3e (max rel)" % rel.max()


# --------------------------------------------------------------------------- #
# Loop wiring: the simulator's forward factor + the loop's pdep hook
# --------------------------------------------------------------------------- #

def test_forward_A_TP_uses_falloff():
    """The reactor's forward factor is k(T,P) (the Falloff) when one is
    attached, and the HPL A(T) otherwise (job-07/step-06 loop deliverable:
    the simulate step uses k(T,P) at the reactor (T,P))."""
    from rmgpu.reactor.simulator import RateParam, forward_A_T, forward_A_TP
    from rmgpu.kinetics.models import Chebyshev

    T, P = 600.0, 1.0e5
    # A hand-built Chebyshev Falloff with known coefficients (SI s^-1).
    coeffs = np.zeros((6, 4))
    coeffs[0, 0] = 5.0  # log10(A) ~ 5 -> A ~ 1e5
    falloff = Chebyshev(coeffs=coeffs, kunits="s^-1",
                        Tmin=300.0, Tmax=800.0, Pmin=1.0e3, Pmax=1.0e6)
    k_tp = falloff.get_rate_coefficient(T, P)

    rp_falloff = RateParam(A=1.0, n=0.0, Ea=0.0, T0=1.0, dS=0.0, falloff=falloff)
    rp_hpl = RateParam(A=1.0, n=0.0, Ea=0.0, T0=1.0, dS=0.0)

    assert forward_A_TP(rp_falloff, T, P) == pytest.approx(k_tp, rel=1e-12), \
        "forward_A_TP must return the Falloff's k(T,P) when one is attached"
    # HPL: forward_A_TP falls back to A_j(T) (P ignored for an Arrhenius).
    assert forward_A_TP(rp_hpl, T, P) == pytest.approx(forward_A_T(rp_hpl, T),
                                                       rel=1e-12), \
        "forward_A_TP must return the HPL A(T) when no Falloff is attached"
    assert forward_A_TP(rp_hpl, T, P) == pytest.approx(1.0, rel=1e-12)


def _minimal_loop_ctx(pressure_dependence=None, pdep_reactions=None):
    """A minimal RunContext (enlarge/ML/families are not exercised here - only
    the pdep hook), for driving CoreEdgeLoop._pdep_update directly."""
    from rmgpu.core.loop import RunContext, LoopConfig
    return RunContext(
        databases=None, ml=None, families=None,
        seed_species=[], initial_mole_fractions={},
        temperature=1000.0, pressure=1.0e5,
        config=LoopConfig(),
        pressure_dependence=pressure_dependence,
        pdep_reactions=pdep_reactions,
    )


def test_pdep_update_attaches_falloff(pdep_network, reference_data):
    """The loop's _pdep_update hook (the HPL-stub replacement) runs the pdep
    driver for the registered networks and attaches the fitted Falloff to the
    reactions' RateParams. Reactions not in the registry keep their HPL rates.
    A network is solved once and re-solved only when its reaction set changes
    (the RMG invalid-network marking, collapsed to a signature check)."""
    from rmgpu.core.loop import CoreEdgeLoop
    from rmgpu.reactor.simulator import RateParam
    from rmgpu.schemas.input import PressureDependenceBlock
    from rmgpu.units import Quantity
    from rmgpu.kinetics.models import Chebyshev, PDepArrhenius

    Tlist, Plist, K_ref, net_reactions, _ = reference_data
    # The YAML pressure_dependence block (drives the grid + method + model).
    block = PressureDependenceBlock(
        method="strong collision",  # normalizes to CSE
        Tmin=Quantity(300.0, "K"), Tmax=Quantity(800.0, "K"), Tcount=8,
        Pmin=Quantity(1.0e3, "Pa"), Pmax=Quantity(1.0e6, "Pa"), Pcount=6,
        interpolation_model="Chebyshev",
    )
    # Registry: reaction_key -> (PDepNetwork, PDepReaction). Map two distinct
    # net reactions of the 2-isomer network to two reactions.
    prxn_iso_fwd = pdep_network.reactions[0]   # a (from,to) net reaction
    prxn_diss_fwd = pdep_network.reactions[-1]  # another
    registry = {
        "A>>A2": (pdep_network, prxn_iso_fwd),
        "A2>>BC": (pdep_network, prxn_diss_fwd),
    }
    loop = CoreEdgeLoop(_minimal_loop_ctx(pressure_dependence=block,
                                          pdep_reactions=registry))
    # The loop's reactions (the RateParams the Falloff attaches to).
    rp_iso = RateParam(A=1.0, n=0.5, Ea=70000.0, T0=1.0, dS=0.0, source="ml")
    rp_diss = RateParam(A=1.0, n=0.5, Ea=60000.0, T0=1.0, dS=0.0, source="ml")
    rp_other = RateParam(A=1.0, n=0.0, Ea=50000.0, T0=1.0, dS=0.0, source="ml")
    loop.reactions["A>>A2"] = {"rp": rp_iso, "react_keys": [], "prod_keys": [],
                               "family": "Isomerization"}
    loop.reactions["A2>>BC"] = {"rp": rp_diss, "react_keys": [], "prod_keys": [],
                                "family": "Dissociation"}
    loop.reactions["OTHER"] = {"rp": rp_other, "react_keys": [], "prod_keys": [],
                               "family": "H_Abstraction"}  # NOT in the registry

    loop._pdep_update(iteration=1)

    # The two registered reactions now carry a fitted Falloff (the HPL stub is
    # replaced); their source is tagged 'pdep'. The unregistered reaction keeps
    # its HPL rate (no Falloff).
    assert isinstance(rp_iso.falloff, (Chebyshev, PDepArrhenius)), \
        "_pdep_update must attach a Falloff to a registered pdep reaction"
    assert isinstance(rp_diss.falloff, (Chebyshev, PDepArrhenius))
    assert rp_iso.source == "pdep"
    assert rp_diss.source == "pdep"
    assert rp_other.falloff is None, \
        "an unregistered reaction must keep its HPL rate (no Falloff)"
    assert rp_other.source == "ml"
    # The driver's wall time was recorded (the 'leverage GPUs' number).
    assert loop._pdep_wall_s > 0.0

    # Re-running with an UNCHANGED reaction set is a no-op (the signature
    # check) - the Falloff is not re-fit, and the wall time is unchanged.
    wall_after_first = loop._pdep_wall_s
    loop._pdep_update(iteration=2)
    assert loop._pdep_wall_s == wall_after_first, \
        "_pdep_update must not re-solve an unchanged network (signature check)"


def test_pdep_update_noop_without_block(pdep_network, reference_data):
    """The loop's _pdep_update hook is a no-op when there is no
    pressure_dependence block or no network registry (the production state
    builder lands in the gate, step 07) - HPL rates stand in, no crash."""
    from rmgpu.core.loop import CoreEdgeLoop
    from rmgpu.reactor.simulator import RateParam

    # No block at all.
    loop = CoreEdgeLoop(_minimal_loop_ctx())
    rp = RateParam(A=1.0, n=0.0, Ea=0.0, T0=1.0, dS=0.0, source="ml")
    loop.reactions["A>>A2"] = {"rp": rp, "react_keys": [], "prod_keys": [],
                               "family": "X"}
    loop._pdep_update(iteration=1)  # must be a silent no-op
    assert rp.falloff is None
    assert rp.source == "ml"
    assert loop._pdep_wall_s == 0.0

    # A block but no registry.
    from rmgpu.schemas.input import PressureDependenceBlock
    from rmgpu.units import Quantity
    block = PressureDependenceBlock(
        method="strong collision",
        Tmin=Quantity(300.0, "K"), Tmax=Quantity(800.0, "K"), Tcount=8,
        Pmin=Quantity(1.0e3, "Pa"), Pmax=Quantity(1.0e6, "Pa"), Pcount=6,
    )
    loop2 = CoreEdgeLoop(_minimal_loop_ctx(pressure_dependence=block))
    rp2 = RateParam(A=1.0, n=0.0, Ea=0.0, T0=1.0, dS=0.0, source="ml")
    loop2.reactions["A>>A2"] = {"rp": rp2, "react_keys": [], "prod_keys": [],
                                "family": "X"}
    loop2._pdep_update(iteration=1)  # no-op: no registry
    assert rp2.falloff is None
    assert loop2._pdep_wall_s == 0.0
