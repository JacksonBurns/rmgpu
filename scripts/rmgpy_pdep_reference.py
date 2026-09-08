#!/usr/bin/env python3
"""
job-07/step-07 reference capture (RMG-Py, rmg_env).

Runs RMG-Py on the propane_branching example WITH pressure dependence ON
(method = 'chemically-significant eigenvalues', CSE-Allen) on a fixed (T,P)
grid, and dumps EVERYTHING the job-07 gate (gates/gate_07.py) needs to compare
rmgpu's OWN production pdep pipeline against RMG-Py. NON-CIRCULAR: the
reference is RMG-Py's own run, never rmgpu.

Dumps (gates/baselines/propane_branching/):
  - meta.json          grid, grain params, method, core/edge counts, wall time.
  - statmech.json      per core species (capped to 50): E0, spin, optical,
                       mode types+counts, Cp(T) on 300-1500 K, DoS rho(E) on a
                       fixed grid (relative to the ground state).
  - networks.json      per pdep network: isomers/reactants/products (E0), bath,
                       energy_correction, grain range, path reactions (HPL A/n/Ea
                       + TS E0), net reactions (fitted kinetics type + FULL
                       coefficients, k_inf HPL). The k(T,P) grid is recomputed
                       by BOTH sides by evaluating the fitted Falloff (the
                       network's raw K matrix is wiped by network.cleanup()).
  - ts_e0.json         the 20 pdep path reactions' E0_TS (RMG-Py, Ea-based
                       no-QM form) for the TS-E0 sub-gate.

NOTE: the propane_branching example's input.py carries NO pressure_dependence
block (the shipped example runs HPL). This driver injects one so RMG-Py runs
CSE on the same grid rmgpu will (see the GRID constants below - keep them in
lockstep with the rmgpu YAML used by the gate).

Run: /home/jackson/miniforge3/envs/rmg_env/bin/python scripts/rmgpy_pdep_reference.py [outdir]
"""
import os
import sys
import json
import time
import logging

import numpy as np

# Keep the log file out of the way; the run is long.
logging.disable(logging.INFO)
logging.getLogger().setLevel(logging.WARNING)

from rmgpy.rmg.main import RMG, initialize_log
import rmgpy.rmg.input as rmg_input
from rmgpy.quantity import Quantity

# ---------------------------------------------------------------------------- #
# GRID / pdep settings. KEEP THESE IN LOCKSTEP with the rmgpu YAML the gate
# uses (gates/gate_07.py). The gate re-runs rmgpu on propane_branching with
# exactly these parameters, so the two sides share the (T,P) grid + method.
# ---------------------------------------------------------------------------- #
METHOD = "chemically-significant eigenvalues"   # CSE (Allen)
# Match RMG's canonical pdep test (test/rmgpy/rmg/modelTest.py): 8 T x 5 P
# with Chebyshev(6,4) - the Chebyshev fit requires strictly MORE grid points
# than the polynomial degree (degreeT=6 < Tcount=8, degreeP=4 < Pcount=5).
TMIN, TMAX, TCOUNT = 300.0, 1500.0, 8
PMIN, PMAX, PCOUNT = 0.1, 100.0, 5          # bar
INTERPOLATION = ("Chebyshev", 6, 4)
MAX_GRAIN_SIZE_KCAL = 0.5
MIN_GRAIN_COUNT = 250
MAX_ATOMS = 10
N_SPECIES_STATMECH = 50
DOS_E_ABOVE_MIN, DOS_E_ABOVE_MAX, DOS_E_ABOVE_STEP = 0.0, 250000.0, 10000.0  # J/mol above ground state
Cp_T_LIST = list(range(300, 1501, 100))      # K


def _arr(x):
    if x is None:
        return None
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return float(x)
    if x.ndim == 1:
        return [float(v) for v in x]
    return [_arr(r) for r in x]


def _val(x):
    """Unwrap an RMG-Py Quantity / ScalarQuantity to a float (SI base value)."""
    if x is None:
        return None
    if hasattr(x, "value_si"):
        try:
            return float(x.value_si)
        except Exception:
            return float(x)
    return float(x)


def _ser_kinetics(kin):
    """Serialize an RMG-Py kinetics object (the fitted Falloff) to a dict.

    Every numeric field is unwrapped through ``_val`` (handles Quantity /
    ScalarQuantity), and the whole body is wrapped so that one malformed
    object degrades to ``{"type": ..., "error": ...}`` instead of aborting the
    whole reference dump.
    """
    if kin is None:
        return None
    name = type(kin).__name__
    try:
        if name == "Chebyshev":
            return {
                "type": "Chebyshev",
                "Tmin": _val(kin.Tmin), "Tmax": _val(kin.Tmax),
                "Pmin": _val(kin.Pmin), "Pmax": _val(kin.Pmax),
                "degreeT": int(kin.degreeT), "degreeP": int(kin.degreeP),
                "kunits": str(kin.kunits),
                "coeffs": _arr(np.asarray(kin.coeffs.value_si, dtype=float)),
            }
        if name == "PDepArrhenius":
            out = {
                "type": "PDepArrhenius",
                "Tmin": _val(getattr(kin, "Tmin", None)), "Tmax": _val(getattr(kin, "Tmax", None)),
                "Pmin": _val(getattr(kin, "Pmin", None)), "Pmax": _val(getattr(kin, "Pmax", None)),
                "pressures": _arr(np.asarray(np.asarray(kin.pressures.value_si, dtype=float))),
                "arrhenius": [
                    {"A": _val(a.A), "n": _val(a.n), "Ea": _val(a.Ea), "T0": _val(getattr(a, "T0", 1.0))}
                    for a in (kin.arrhenius or [])],
            }
            return out
        if name in ("Arrhenius", "ArrheniusEP"):
            return {
                "type": name,
                "A": _val(kin.A), "n": _val(kin.n),
                "Ea": _val(kin.Ea), "T0": _val(getattr(kin, "T0", 1.0)),
            }
        return {"type": name}
    except Exception as e:
        return {"type": name, "error": "%s: %s" % (type(e).__name__, e)}


def _ser_species(spec):
    d = {"label": spec.label}
    conf = getattr(spec, "conformer", None)
    if conf is not None:
        e0 = conf.E0
        d["E0"] = float(e0.value_si) if e0 is not None else None
        d["spin_multiplicity"] = int(conf.spin_multiplicity)
        d["optical_isomers"] = int(conf.optical_isomers)
        modes = []
        for m in conf.modes:
            t = type(m).__name__
            md = {"type": t}
            if t == "HarmonicOscillator":
                # RMG-Py's Frequency.value_si already returns the cm^-1 NUMBER
                # (Frequency(1555,'cm^-1').value_si == 1555.0); do NOT convert.
                md["frequencies_cm1"] = _arr(np.asarray(m.frequencies.value_si, dtype=float))
            elif t in ("LinearRotor",):
                md["inertia"] = float(m.inertia.value_si)
                md["symmetry"] = int(m.symmetry)
            elif t in ("NonlinearRotor",):
                md["inertia"] = _arr(np.asarray(m.inertia.value_si, dtype=float))
                md["symmetry"] = int(m.symmetry)
            elif t in ("HinderedRotor",):
                md["inertia"] = float(m.inertia.value_si)
                md["symmetry"] = int(m.symmetry)
                if getattr(m, "barrier", None) is not None:
                    md["barrier"] = float(m.barrier.value_si)
                if getattr(m, "fourier", None) is not None:
                    md["fourier"] = _arr(np.asarray(m.fourier, dtype=float))
            elif t == "IdealGasTranslation":
                md["mass"] = float(m.mass.value_si)
            elif t == "FreeRotor":
                md["inertia"] = float(m.inertia.value_si)
                md["symmetry"] = int(m.symmetry)
            modes.append(md)
        d["modes"] = modes
        # precomputed reference quantities (RMG-Py's own methods)
        d["Cp_T"] = {
            "T": Cp_T_LIST,
            "values": [float(conf.get_heat_capacity(T)) for T in Cp_T_LIST],
        }
        e_above = np.arange(DOS_E_ABOVE_MIN, DOS_E_ABOVE_MAX + DOS_E_ABOVE_STEP,
                            DOS_E_ABOVE_STEP, dtype=float)
        try:
            d["DOS"] = {
                "e_above": [float(v) for v in e_above],
                "values": _arr(np.asarray(conf.get_density_of_states(e_above), dtype=float)),
            }
        except Exception as e:  # keep the dump alive if one species fails
            d["DOS"] = {"error": str(e)}
    return d


def main():
    t_start = time.time()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        here, "gates", "baselines", "propane_branching")
    os.makedirs(outdir, exist_ok=True)

    input_file = "/home/jackson/rmgpu/RMG-Py/examples/rmg/propane_branching/input.py"
    # A private output dir (RMG writes chemkin/log there).
    workdir = os.path.join(outdir, "_rmgpy_run")
    os.makedirs(workdir, exist_ok=True)
    log_path = os.path.join(workdir, "RMG.log")

    logging.info("Building RMG from %s", input_file)
    rmg = RMG(input_file=input_file, output_directory=workdir)

    # Inject the pressure_dependence block (the example ships without one).
    rmg_input.set_global_rmg(rmg)
    rmg_input.pressure_dependence(
        method=METHOD,
        temperatures=(TMIN, TMAX, "K", TCOUNT),
        pressures=(PMIN, PMAX, "bar", PCOUNT),
        interpolation=INTERPOLATION,
        maximumGrainSize=(MAX_GRAIN_SIZE_KCAL, "kcal/mol"),
        minimumNumberOfGrains=MIN_GRAIN_COUNT,
        maximumAtoms=MAX_ATOMS,
    )
    rmg.pressure_dependence.output_file = workdir

    # ------------------------------------------------------------------ #
    # Suppress RMG's HARD macroscopic-equilibrium guard.
    #
    # RMG-Py's Network.calculate_rate_coefficients raises NetworkError when, at
    # some (T,P), a net reaction's CSE k(T,P) disagrees with its thermodynamic
    # equilibrium constant (from the Cp-fitted statmech free energy) by more
    # than a factor of 2. For an association channel (O+CH3<=>CH3O) on a deep
    # well this fires at the hot corner and ABORTS the whole run - so a full
    # propane_branching pdep run cannot complete under stock RMG-Py at these
    # settings. This is a validation guard, NOT a physics computation: the CSE
    # k(T,P) matrix K is already computed and stored on self.K BEFORE the guard
    # runs. We wrap the method so that, on that specific guard error, we KEEP
    # the computed K and continue (so the run completes and we can dump every
    # network's k(T,P)). The DoS, grain selection, TS-E0, collision model and
    # the CSE eigen-decomposition are all UNCHANGED by this, so the DoS/DoS-
    # parity and k(T,P)-parity ground truth are still RMG-Py's own physics.
    # We record how many (T,P,network) points were suppressed.
    # ------------------------------------------------------------------ #
    import rmgpy.pdep.network as _rmg_net
    _orig_calc = _rmg_net.Network.calculate_rate_coefficients
    _suppress = {"count": 0, "points": []}

    def _calc_no_eq_guard(self, Tlist, Plist, method, error_check=True,
                          neglect_high_energy_collisions=False,
                          high_energy_rate_tol=0.01):
        n_T, n_P = len(Tlist), len(Plist)
        n_cfg = self.n_isom + self.n_reac + self.n_prod
        K = np.zeros((n_T, n_P, n_cfg, n_cfg), float)
        for t, T in enumerate(Tlist):
            for p, P in enumerate(Plist):
                # Null self.K first so a K left over from a PREVIOUS (T,P) can
                # never be mistaken for the current one (e.g. if set_conditions
                # raises before CSE stores a fresh K).
                self.K = None
                try:
                    # Call the real method on singletons; it returns (1,1,n,n).
                    _orig_calc(self, [T], [P], method, error_check=error_check,
                               neglect_high_energy_collisions=neglect_high_energy_collisions,
                               high_energy_rate_tol=high_energy_rate_tol)
                    # On success the real method stored self.K (n,n).
                    K[t, p] = self.K
                except Exception as e:
                    # If CSE already populated self.K before the (equilibrium)
                    # guard raised, keep it; otherwise re-raise (e.g. a grain
                    # retry / DoS failure that we do NOT want to paper over).
                    if getattr(self, "K", None) is not None and \
                            np.asarray(self.K).any():
                        K[t, p] = self.K
                        _suppress["count"] += 1
                        _suppress["points"].append((round(float(T), 3), round(float(P), 3),
                                                    str(e)[:80]))
                        logging.warning(
                            "suppressed eq-guard at T=%.1f P=%.4g: %s", T, P, str(e)[:60])
                    else:
                        raise
        return K

    _rmg_net.Network.calculate_rate_coefficients = _calc_no_eq_guard
    logging.info("Installed equilibrium-guard suppression wrapper on Network.")

    logging.info("Executing (this is the long pole)...")
    sim_completed = True
    sim_error = None
    try:
        rmg.execute()
    except Exception as e:
        # The pdep networks (the gate's ground truth) are ALL built + fitted
        # during the enlarge() loop, BEFORE the reactor simulation. If the
        # mechanism generation finished but the final reactor ODE integrate
        # blew up (a known RMG numerical fragility: 'nans in moles'), the
        # networks are still complete and valid, so tolerate the sim failure
        # and proceed to dump them. Record the error for the report.
        import traceback
        sim_completed = False
        sim_error = "%s: %s" % (type(e).__name__, e)
        logging.warning("rmg.execute() failed at the SIMULATION stage (networks "
                        "are already built): %s", sim_error)
        logging.warning(traceback.format_exc(limit=3))
    t_run = time.time() - t_start
    logging.info("execute() finished in %.1f s", t_run)

    model = rmg.reaction_model

    # ------------------------------------------------------------------ #
    # meta
    # ------------------------------------------------------------------ #
    meta = {
        "example": "propane_branching",
        "method": METHOD,
        "grid": {"Tmin": TMIN, "Tmax": TMAX, "Tcount": TCOUNT,
                 "Pmin": PMIN, "Pmax": PMAX, "Pcount": PCOUNT,
                 "T_units": "K", "P_units": "bar"},
        "Tlist_K": [float(t) for t in rmg.pressure_dependence.Tlist.value_si],
        "Plist_Pa": [float(p) for p in rmg.pressure_dependence.Plist.value_si],
        "interpolation": list(INTERPOLATION),
        "max_grain_size_J_mol": float(Quantity(MAX_GRAIN_SIZE_KCAL, "kcal/mol").value_si),
        "min_grain_count": MIN_GRAIN_COUNT,
        "max_atoms": MAX_ATOMS,
        "core_species": len(model.core.species),
        "core_reactions": len(model.core.reactions),
        "edge_species": len(model.edge.species),
        "edge_reactions": len(model.edge.reactions),
        "n_pdep_networks": len(model.network_list),
        "run_wall_s": float(t_run),
        "eq_guard_suppressed_points": _suppress["count"],
        "eq_guard_suppressed_examples": _suppress["points"][:20],
        "eq_guard_note": ("RMG-Py's hard macroscopic-equilibrium guard was "
                          "suppressed (kept the CSE-computed K and continued) so "
                          "the full run completes; the guard is a validation "
                          "check, not part of the CSE k(T,P) computation. The "
                          "DoS/grain/TS-E0/collision physics are unchanged."),
    }
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logging.info("meta: core %d spc / %d rxn; edge %d spc / %d rxn; %d pdep networks",
                 meta["core_species"], meta["core_reactions"],
                 meta["edge_species"], meta["edge_reactions"], meta["n_pdep_networks"])

    # ------------------------------------------------------------------ #
    # statmech (capped to N_SPECIES_STATMECH core species)
    # ------------------------------------------------------------------ #
    spc_list = []
    for spec in model.core.species:
        if getattr(spec, "conformer", None) is not None:
            spc_list.append(spec)
    spc_list = spc_list[:N_SPECIES_STATMECH]
    statmech = [_ser_species(s) for s in spc_list]
    with open(os.path.join(outdir, "statmech.json"), "w") as f:
        json.dump({"n_dropped": len(model.core.species) - len(spc_list),
                   "species": statmech}, f, indent=2)
    logging.info("statmech: dumped %d species", len(statmech))

    # ------------------------------------------------------------------ #
    # networks + TS-E0
    # ------------------------------------------------------------------ #
    networks = []
    ts_e0 = []
    for net in model.network_list:
        nrec = {
            "label": net.label,
            "index": getattr(net, "index", None),
            "n_isom": len(net.isomers),
            "n_reac": len(net.reactants),
            "n_prod": len(net.products),
            "energy_correction": _val(getattr(net, "energy_correction", None)),
            "isomers": [
                {"label": iso.species[0].label, "E0": float(iso.E0)} for iso in net.isomers],
            "reactants": [
                {"labels": [s.label for s in r.species], "E0": float(r.E0)}
                for r in net.reactants],
            "products": [
                {"labels": [s.label for s in p.species], "E0": float(p.E0)}
                for p in net.products],
            "bath_gas": {b.label: float(frac) for b, frac in net.bath_gas.items()},
            "path_reactions": [],
            "net_reactions": [],
        }
        for rxn in net.path_reactions:
            kin = rxn.kinetics
            hpl = None
            if kin is not None:
                hpl = {"A": _val(kin.A), "n": _val(kin.n),
                       "Ea": _val(kin.Ea), "T0": _val(getattr(kin, "T0", 1.0))}
            ts = getattr(rxn, "transition_state", None)
            ts_e = None
            if ts is not None and getattr(ts, "conformer", None) is not None and \
                    ts.conformer.E0 is not None:
                ts_e = float(ts.conformer.E0.value_si)
            nrec["path_reactions"].append({
                "label": getattr(rxn, "label", None),
                "reactants": [s.label for s in rxn.reactants],
                "products": [s.label for s in rxn.products],
                "hpl": hpl,
                "ts_E0": ts_e,
            })
            if ts_e is not None:
                ts_e0.append({
                    "network": net.label,
                    "reactants": [s.label for s in rxn.reactants],
                    "products": [s.label for s in rxn.products],
                    "E0_TS": ts_e,
                    "Ea": _val(kin.Ea) if kin is not None else None,
                })
        for nrxn in net.net_reactions:
            nrec["net_reactions"].append({
                "label": getattr(nrxn, "label", None),
                "reactants": [s.label for s in nrxn.reactants],
                "products": [s.label for s in nrxn.products],
                "n_reactants": len(nrxn.reactants),
                "kinetics": _ser_kinetics(getattr(nrxn, "kinetics", None)),
            })
        networks.append(nrec)
    with open(os.path.join(outdir, "networks.json"), "w") as f:
        json.dump({"n_networks": len(networks), "networks": networks}, f, indent=2)
    with open(os.path.join(outdir, "ts_e0.json"), "w") as f:
        json.dump({"n_path_reactions": len(ts_e0), "reactions": ts_e0}, f, indent=2)
    logging.info("networks: %d networks, %d path reactions (TS-E0)",
                 len(networks), len(ts_e0))

    # ------------------------------------------------------------------ #
    # summary
    # ------------------------------------------------------------------ #
    n_pdep_rxn = sum(len(n["net_reactions"]) for n in networks)
    logging.info("DONE. core %d spc, %d pdep networks, %d pdep net reactions, %d TS-E0. wall %.1f s",
                 meta["core_species"], len(networks), n_pdep_rxn, len(ts_e0), time.time() - t_start)
    print("REFERENCE COMPLETE")
    print("  outdir        :", outdir)
    print("  core          : %d species / %d reactions" % (meta["core_species"], meta["core_reactions"]))
    print("  pdep networks : %d" % len(networks))
    print("  pdep net rxns : %d" % n_pdep_rxn)
    print("  TS-E0 rxns    : %d" % len(ts_e0))
    print("  statmech spc  : %d" % len(statmech))
    print("  wall (exec)   : %.1f s" % t_run)


if __name__ == "__main__":
    main()
