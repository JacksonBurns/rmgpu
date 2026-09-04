#!/usr/bin/env python3
"""Job-06 gate: first real mechanism generation (the moment of truth).

Runs the checks named in the job-06 brief and records a machine-readable
results JSON (reports/gate_06_results.json). It is NON-circular: the parity
reference (RMG-Py, pdep OFF) is the real RMG-Py output committed under
gates/baselines/<example>/ by scripts/rmgpy_reference.py, not rmgpu's own
output.

This is the HONEST gate (job-06/step-07: the original was closed GREEN on
false evidence). Checks:
  (1) torchdae stiff-ODE sub-gate (VdP + Lindemann toy vs RK45) - hard.
  (2) superminimal `rmgpu run` completes; parity vs RMG-Py is a RECORDED
      QUALITATIVE FINDING (set diffs + documented cause), NOT a hard check -
      per the user's 2026-09-03 parity bar, a divergent core with a documented
      cause is a PASS.
  (3) c3h4 `rmgpu run` with the GRI-Mech3 seed mechanism ACTUALLY loaded:
      hard checks = run completes + core species count well above the 3 seed
      species (floor 30; the GRI seed is 54) + a non-zero seed reaction count
      + output tree + physically valid final profile. Parity vs RMG-Py is a
      recorded finding, not a hard check.
  (4) Output tree: every PLAN 12.3 file exists and is valid in BOTH examples;
      core.yaml loads back via the schema.
  (5) PHYSICAL VALIDITY (new, both examples): every mole fraction in the
      written profile rows is in [0, 1] (eps 1e-6) and each row sums to 1
      within 1e-3. A non-physical final profile is a HARD FAIL - a blow-up is
      never "qualitatively similar" to a reference.
  (6) Provenance: real hashes/versions in BOTH examples.

Hard checks (exit 1 if any fails):
  subgate, superminimal run completes, c3h4 status PASS, output tree PASS
  (both), physical validity PASS (both), provenance PASS (both).

The re-simulation of core.yaml's final mechanism is RECORDED but NOT a hard
check (it re-simulates the full core from the initial state, which is a
different, longer-time problem than the loop's last snapshot simulation -
see _resimulate_final's note).
"""
import csv
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, "gates", "baselines")
OUT = os.path.join(REPO, "reports", "gate_06_results.json")
RUN_SUPERMIN = os.path.join(REPO, "examples", "run_output_sm")
RUN_C3H4 = os.path.join(REPO, "examples", "run_output_c3h4")
RMGPU_EXAMPLES = os.path.join(REPO, "examples")

# Floor for the c3h4 core species count: the 3 input seed species are CH2,
# C2H2, N2; the GRI-Mech3 seed mechanism adds 54 library species, so a real
# seed load yields a core far above 3. Floor 30 is a sanity check that the
# seed (not a 3-species stub) was loaded.
C3H4_CORE_SPECIES_FLOOR = 30
# The c3h4 run is LONG (full GRI seed, 1350 K, ~51 families x ~1275 core-pairs
# of generate_reactions per iteration, ~300 s/iteration; a 25-iteration run is
# hours). Per job-06/step-07 we bound it: the seed is demonstrably loaded and
# the profile is the hard requirement, so we cap the iterations and RECORD the
# iteration count + core/edge counts reached.
C3H4_MAX_ITERATIONS = 1
# Physical validity tolerances (job-06/step-07 Fix 5).
PHYS_EPS = 1.0e-6     # mole fractions must be in [0-eps, 1+eps]
PHYS_SUM_TOL = 1.0e-3  # each row (closed system) must sum to 1 within this

# PLAN 12.3 required output-tree files (relative to the run root).
PLAN123_FILES = [
    "run.yaml", "provenance.yaml", "summary.md", "rmgpu.log", "events.jsonl",
    "mechanism/core.yaml", "mechanism/edge.yaml",
    "mechanism/chem.inp", "mechanism/chem_annotated.inp",
    "mechanism/species_dictionary.txt",
    "reactions/reactions.json",
    "profiles/reactor/time_series.csv", "profiles/reactor/metadata.yaml",
]


# ---------------------------------------------------------------------------
# canonical keys (identical to rmgpu.core.loop.canonical_key and to
# scripts/rmgpy_reference.py: AddHs if implicit, kekulize, sanitize, canonical
# SMILES). Both sides of the parity are keyed this way.
# ---------------------------------------------------------------------------
def canon_smiles(s):
    from rdkit import Chem
    m = Chem.MolFromSmiles(s)
    if m is None:
        return s
    if not any(a.GetSymbol() == "H" for a in m.GetAtoms()):
        try:
            m = Chem.AddHs(m)
        except Exception:
            pass
    try:
        Chem.Kekulize(m, clearAromaticFlags=True)
    except Exception:
        pass
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    return Chem.MolToSmiles(m)


def load_core_artifact(path):
    """Load a mechanism core.yaml -> (label_to_canon, {label}, {rxn key})."""
    import yaml
    art = yaml.safe_load(open(path))
    core = art["core"]
    label_to_canon = {}
    for sp in core["species"]:
        label_to_canon[sp["label"]] = canon_smiles(sp["smiles"])
    canon_set = set(label_to_canon.values())
    return label_to_canon, canon_set


def artifact_seed_reaction_count(run_root):
    """Count family=='seed' reactions in the written core.yaml + edge.yaml
    (the artifact is the ground truth for whether a seed mechanism was
    actually loaded and written)."""
    import yaml
    n = 0
    for fname in ("core.yaml", "edge.yaml"):
        p = os.path.join(run_root, "mechanism", fname)
        if not os.path.exists(p):
            continue
        art = yaml.safe_load(open(p))
        core = art.get("core") or {}
        for r in core.get("reactions") or []:
            if (r.get("family") or "") == "seed":
                n += 1
    return n


def setdiff_report(name, a, b):
    """a = rmgpu set, b = reference set. Returns a dict of the diffs."""
    return {
        "rmgpu_count": len(a),
        "reference_count": len(b),
        "shared": sorted(a & b),
        "only_in_rmgpu": sorted(a - b),
        "only_in_reference": sorted(b - a),
        "identical": a == b,
    }


# ---------------------------------------------------------------------------
# Check 1: torchdae stiff-ODE sub-gate
# ---------------------------------------------------------------------------
def check_subgate():
    out = {"status": "pending"}
    # (a) non-chemistry control: Van der Pol mu=10 vs RK45
    from rmgpu.reactor.torch import validate_stiff_ode
    vdp = validate_stiff_ode(mu=10.0, t_end=10.0)
    # (b) chemistry control: 3-reaction Lindemann-falloff toy, H2 + I2 -> 2 HI
    lind = _lindemann_subgate()
    out.update({
        "van_der_pol_mu10_max_abs_diff": vdp,
        "lindemann_toy_max_abs_diff": lind["max_abs_diff"],
        "lindemann_toy_endstate": lind["endstate"],
    })
    ok = vdp < 1.0 and lind["max_abs_diff"] < 1.0
    out["status"] = "PASS" if ok else "FAIL"
    return out


def _lindemann_subgate():
    """A 3-reaction Lindemann-falloff toy as a closed mole-fraction ODE (the
    same formulation the real simulator uses: dy_i/dt = sum_j nu_ij v_j with
    v_j a 1/s net rate), integrated by torchdae TR-BDF2 (fixed h) and compared
    to a tiny-step RK45 reference. The point (PLAN 13 risk 3) is to prove the
    backend handles a STIFF, multi-rate chemistry ODE (fast H2+I->HI chain on a
    slow I2->2I source) to within a tight tolerance before trusting it to drive
    mechanism growth."""
    import torch
    import torchdae

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # stiff separation: slow I2 source (k0 ~ 1/s), fast I-consuming chain (k1 ~ 50)
    k0 = 1.0    # I2 -> I + I   (third-body, M=1)      v0 = k0*y_I2
    k1 = 50.0   # H2 + I -> 2 HI                        v1 = k1*y_H2*y_I
    k2 = 5.0    # I2 + I -> 2 HI                        v2 = k2*y_I2*y_I
    order = ["I2", "I", "H2", "HI"]  # index 0..3

    def dydt(y):
        y1 = y if y.dim() == 1 else y[0]
        y_I2 = torch.clamp(y1[0], min=0.0)
        y_I = torch.clamp(y1[1], min=0.0)
        y_H2 = torch.clamp(y1[2], min=0.0)
        v0 = k0 * y_I2
        v1 = k1 * y_H2 * y_I
        v2 = k2 * y_I2 * y_I
        out = torch.stack([-v0 - v2, 2 * v0 - v1 - v2, -v1, 2 * v1 + 2 * v2])
        if y.dim() == 2:
            out = out[None, :]
        return out

    y0 = torch.tensor([0.5, 0.0, 0.5, 0.0], dtype=torch.float64, device=device)
    t_end = 2.0
    h = 1.0e-4
    y0_t = y0[None, :]

    def F(t, y, yp):
        return yp - dydt(y)

    yp0 = dydt(y0_t)  # consistent 2-D initial derivative
    sol = torchdae.solve_tr_bdf2(F, (0.0, t_end), y0_t, h=h, yp0=yp0)
    ts = sol.ts.cpu().numpy()
    ys = sol.ys.squeeze(1).cpu().numpy()  # (n, 4)

    # tiny-step RK45 reference
    from scipy.integrate import solve_ivp

    def rhs(t, yv):
        y_I2, y_I, y_H2, y_HI = [max(0.0, x) for x in yv]
        v0 = k0 * y_I2
        v1 = k1 * y_H2 * y_I
        v2 = k2 * y_I2 * y_I
        return [-v0 - v2, 2 * v0 - v1 - v2, -v1, 2 * v1 + 2 * v2]

    ref = solve_ivp(rhs, (0.0, t_end), [0.5, 0.0, 0.5, 0.0], method="RK45",
                    rtol=1e-10, atol=1e-13, max_step=1e-5, t_eval=ts)
    max_diff = float(torch.max(torch.abs(torch.tensor(ys) - torch.tensor(ref.y.T))).item())
    endstate = ys[-1].tolist()
    total = sum(endstate)
    return {"max_abs_diff": max_diff, "endstate": {order[i]: round(endstate[i], 6)
                                                   for i in range(4)},
            "endstate_sum": round(total, 8),
            "note": "stiff 3-rxn Lindemann chain; torchdae TR-BDF2 h=%.0e vs RK45" % h}


# ---------------------------------------------------------------------------
# Physical validity (job-06/step-07 Fix 5) - both examples, HARD.
# ---------------------------------------------------------------------------
def check_physical_validity(run_root, eps=PHYS_EPS, sum_tol=PHYS_SUM_TOL):
    """Read profiles/reactor/time_series.csv and check every row:
      - every mole-fraction value is FINITE and in [-eps, 1+eps];
      - the row sum is within sum_tol of 1.0 (closed mole-fraction vector).
    FAIL (with the offending species/values recorded) if any value is
    non-finite (nan/inf - a blow-up), out of range, or any row sum is off.
    This is what catches a non-physical blow-up (e.g. mole fractions ~250 or
    an inf/nan from an overflowed rate). Do NOT loosen these to force a pass."""
    import math
    out = {"status": "pending", "rows": 0, "cols": 0, "bad_values": [],
           "row_sum_errors": [], "final_row": None, "final_row_sum": None}
    path = os.path.join(run_root, "profiles", "reactor", "time_series.csv")
    if not os.path.exists(path):
        out["status"] = "FAIL"
        out["reason"] = "profiles/reactor/time_series.csv missing"
        return out
    with open(path) as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        out["status"] = "FAIL"
        out["reason"] = "time_series.csv has no data rows"
        return out
    header = rows[0]
    cols = [h.replace("_molefrac", "") for h in header[1:]]
    out["rows"] = len(rows) - 1
    out["cols"] = len(cols)
    bad_values = []
    row_sum_errors = []
    for i, row in enumerate(rows[1:]):
        vals = []
        for j, cell in enumerate(row[1:]):
            try:
                v = float(cell)
            except ValueError:
                bad_values.append({"row": i + 1, "species": cols[j],
                                   "value": cell, "reason": "not a number"})
                continue
            vals.append(v)
            if not math.isfinite(v):
                bad_values.append({"row": i + 1, "species": cols[j],
                                   "value": v, "reason": "non-finite (nan/inf)"})
            elif v < -eps or v > 1.0 + eps:
                bad_values.append({"row": i + 1, "species": cols[j],
                                   "value": v})
        if vals and all(math.isfinite(v) for v in vals):
            s = sum(vals)
            if abs(s - 1.0) > sum_tol:
                row_sum_errors.append({"row": i + 1, "sum": s})
        elif not all(math.isfinite(v) for v in vals):
            # a row with nan/inf is non-physical on its own
            row_sum_errors.append({"row": i + 1, "sum": None,
                                   "reason": "non-finite value in row"})
    out["bad_values"] = bad_values[:50]  # cap the report
    out["row_sum_errors"] = row_sum_errors[:50]
    final = [float(x) for x in rows[-1][1:]]
    out["final_row"] = {cols[j]: final[j] for j in range(len(cols))}
    out["final_row_sum"] = (float(sum(final)) if all(math.isfinite(v) for v in final)
                            else None)
    if bad_values or row_sum_errors:
        out["status"] = "FAIL"
        out["n_bad_values"] = len(bad_values)
        out["n_row_sum_errors"] = len(row_sum_errors)
    else:
        out["status"] = "PASS"
    return out


# ---------------------------------------------------------------------------
# summary.md parser (fast path: the run output already exists)
# ---------------------------------------------------------------------------
def _parse_summary_md(run_root):
    txt = open(os.path.join(run_root, "summary.md")).read()

    def _find(k):
        m = re.search(rf"{k}: (.+)", txt)
        return m.group(1).strip() if m else None

    summary = {
        "iterations": int(_find("Iterations") or 0),
        "steady_state": "steady state" in txt,
        "core_species_count": int(_find("Core species") or 0),
        "core_reaction_count": int(_find("Core reactions") or 0),
        "edge_species_count": int(_find("Edge species") or 0),
        "edge_reaction_count": int(_find("Edge reactions") or 0),
        # REAL estimation/coverage numbers (job-06/step-07 Fix 8: never {})
        "estimation_counts": {
            "library_hits": int(_find("library_hits") or 0),
            "ml_hits": int(_find("ml_hits") or 0),
            "coverage_errors": int(_find("coverage_errors") or 0),
        },
        "coverage": {
            "thermo_errors": int(_find("thermo_errors") or 0),
            "kinetics_errors": int(_find("kinetics_errors") or 0),
            "species_dropped": int(_find("species_dropped") or 0),
            "reactions_dropped": int(_find("reactions_dropped") or 0),
        },
        "blocked_ml": [],
    }
    return summary


# ---------------------------------------------------------------------------
# Check 2 + 3: rmgpu run superminimal + parity (parity = recorded finding)
# ---------------------------------------------------------------------------
def check_superminimal():
    import rmgpu.main as M
    import time
    out = {"status": "pending"}
    core_yaml_path = os.path.join(RUN_SUPERMIN, "mechanism", "core.yaml")
    summary_path = os.path.join(RUN_SUPERMIN, "summary.md")
    if os.path.exists(core_yaml_path) and os.path.exists(summary_path):
        summary = _parse_summary_md(RUN_SUPERMIN)
        out["elapsed_seconds"] = 0.0
        out["skipped_run"] = True
    else:
        t0 = time.time()
        summary = M.run(os.path.join(RMGPU_EXAMPLES, "superminimal.yaml"),
                        out_root=RUN_SUPERMIN)
        out["elapsed_seconds"] = round(time.time() - t0, 1)
        out["skipped_run"] = False
    out.update({
        "iterations": summary["iterations"],
        "steady_state": summary["steady_state"],
        "core_species_count": summary["core_species_count"],
        "core_reaction_count": summary["core_reaction_count"],
        "edge_species_count": summary["edge_species_count"],
        "edge_reaction_count": summary["edge_reaction_count"],
        "estimation_counts": summary["estimation_counts"],
        "coverage_gaps": summary["coverage"],
        "blocked_ml": summary["blocked_ml"],
    })
    # run completes (steady state or max iterations) - a HARD check.
    out["status"] = "PASS" if summary["steady_state"] else "MAX_ITER"

    # --- parity vs the committed RMG-Py baseline: RECORDED FINDING, not hard ---
    base_path = os.path.join(BASE, "superminimal", "summary.json")
    parity = {}
    if os.path.exists(base_path):
        base = json.load(open(base_path))
        lab2c, rmgpu_core_sp = load_core_artifact(core_yaml_path)
        rxnjson = json.load(open(os.path.join(RUN_SUPERMIN, "reactions",
                                              "reactions.json")))
        rmgpu_core_rx = _reaction_keys_from_json(rxnjson, lab2c)
        rmg_core_sp = set(base["core_species"])
        rmg_core_rx = set(base["core_reactions_keys"])
        parity = {
            "core_species": setdiff_report("core_species", rmgpu_core_sp, rmg_core_sp),
            "core_reactions": setdiff_report("core_reactions", rmgpu_core_rx, rmg_core_rx),
        }
    out["parity_vs_rmgpy"] = parity
    return out


def _reaction_keys_from_json(rxnjson, label_to_canon):
    """canonical, order-insensitive reaction keys from rmgpu reactions.json."""
    out = set()
    for r in rxnjson:
        rs = sorted(label_to_canon.get(x, x) for x in r["reactants"])
        ps = sorted(label_to_canon.get(x, x) for x in r["products"])
        out.add(".".join(rs) + ">>" + ".".join(ps))
    return out


def _divergence_cause():
    """The root-cause explanation for the superminimal core divergence vs
    RMG-Py. This is the 'documented cause' (a PASS, not a stack failure) for a
    non-identical core, written to match the ACTUAL recorded run (job-06/
    step-07 Fix 6: the old text described an earlier 70-rxn/11-O-chain run and
    misattributed the fix to job-07)."""
    return {
        "verdict": "DIVERGENT (documented, not a stack failure; parity bar per "
                   "user 2026-09-03: qualitatively similar, documented "
                   "deviations acceptable)",
        "actual_run": "superminimal core 20 species / 100 reactions vs RMG-Py "
                      "13 species / 19 reactions (5 shared reactions); 15 O-chain "
                      "species (H-O_n-H and O_n diradicals, n=2..9) rmgpu-only",
        "core_species_in_rmgpu_only": "15 long O-chain species (H-O_n-H and O_n "
            "diradicals, n=2..9). rmgpu's fixed-timescale rate-ratio screening "
            "promotes them to the core, where RMG-Py keeps them pruned in the edge.",
        "core_species_in_rmgpy_only": "4 inert third-body species (He, Ne, N#N, "
            "[Ar]) that RMG-Py auto-injects into the reactor/core (constant "
            "species), plus atomic O and O2(singlet) which RMG-Py's screening "
            "promotes. rmgpu's HPL-stub reactor adds no inerts (third-body/pdep "
            "is job-07 scope) and its screening does not promote them.",
        "core_reactions": "rmgpu core has 100 reactions including the O-chain "
            "Birad_recombination growth reactions; RMG-Py core has 19. The "
            "over-generation is Birad_recombination producing O-chain products, "
            "amplified by the looser screening criterion.",
        "root_cause_screening": "rmgpu's _screen (rmgpu/core/loop.py) promotes a "
            "species to the core when the max of its reactions' forward/reverse "
            "rate-ratio exceeds tolerance_move_to_core, evaluated on a FIXED-"
            "TIMESCALE SNAPSHOT (t_end = 5.0/char, CHAR_RATE_TFACTOR). It is "
            "promote-only: there is NO rate-ratio demotion - prune() only removes "
            "edge species not referenced by an edge reaction. RMG-Py's screen "
            "(rmgpy/rmg/model.py:1418-1455) uses reactor-driven "
            "max_edge_species_rate_ratios (from the actual reactor solution, not "
            "a fixed snapshot) and prunes below tolerance_keep_in_edge AND keeps "
            "in the edge between keep and move-to-core. So the O-chain diradicals "
            "get promoted and STAY in the rmgpu core while RMG-Py keeps them "
            "pruned. This is a screening-criterion difference (fixed-snapshot "
            "max(fwd,rev), promote-only, no demotion) - NOT a job-07/pdep "
            "artifact. pdep does not stop O-chain diradical recombination "
            "enumeration or the screening promotion.",
        "responsible_modules": [
            "rmgpu/core/loop.py (_screen, prune): fixed-snapshot promote-only screening",
            "rmgpu/reactor/simulator.py: no inert/third-body species (HPL stub; job-07 turns pdep on in both)",
        ],
        "known_rmgpu_behavior": "Yes - more aggressive screening (O-chain "
            "diradicals kept in core). Recorded as an acceptable deviation per "
            "the user's parity bar. Aligning the screening criterion (reactor-"
            "driven rate ratios + keep-in-edge demotion) is a SEPARATE follow-up "
            "step; it is NOT fixed by job-07.",
    }


# ---------------------------------------------------------------------------
# Check 4: c3h4 (seed mechanism MUST actually load - hard)
# ---------------------------------------------------------------------------
def check_c3h4():
    """c3h4 with the GRI-Mech3 seed mechanism. HARD: the run must complete
    with the seed actually loaded (core species >= C3H4_CORE_SPECIES_FLOOR, a
    non-zero seed reaction count in the written artifact), the output tree
    must exist and load back, and the final profile must be physically valid.
    Parity vs RMG-Py is a RECORDED FINDING (expected to diverge - different
    screening), NOT a hard gate."""
    import rmgpu.main as M
    import time
    out = {"status": "pending", "run_ok": False}
    run_root = RUN_C3H4
    core_yaml_path = os.path.join(run_root, "mechanism", "core.yaml")
    if not os.path.exists(core_yaml_path):
        t0 = time.time()
        try:
            summary = M.run(os.path.join(RMGPU_EXAMPLES, "c3h4.yaml"),
                            out_root=run_root,
                            max_iterations=C3H4_MAX_ITERATIONS)
            out["elapsed_seconds"] = round(time.time() - t0, 1)
            out["run_ok"] = True
            out["skipped_run"] = False
            out["run_exception"] = None
        except Exception as e:
            out["status"] = "FAIL"
            out["reason"] = f"rmgpu run failed: {type(e).__name__}: {e}"
            return out
    else:
        out["elapsed_seconds"] = 0.0
        out["skipped_run"] = True
        out["run_ok"] = True
        out["run_exception"] = None
    # Real summary numbers from the written tree (never hardcoded).
    summary = _parse_summary_md(run_root)
    out.update(summary)

    # HARD: seed mechanism actually loaded (from the written artifact, the
    # ground truth - not the run's memory).
    seed_rxn_count = artifact_seed_reaction_count(run_root)
    n_core_species = summary["core_species_count"]
    out["seed_reaction_count_in_artifact"] = seed_rxn_count
    out["core_species_floor"] = C3H4_CORE_SPECIES_FLOOR

    # HARD: output tree exists + core.yaml loads back.
    out["output_tree"] = check_output_tree(run_root, label="c3h4")

    # HARD: physically valid final profile.
    out["physical_validity"] = check_physical_validity(run_root)

    reasons = []
    if not out["run_ok"]:
        reasons.append("run did not complete")
    if n_core_species < C3H4_CORE_SPECIES_FLOOR:
        reasons.append(
            f"seed mechanism not loaded: core species {n_core_species} < floor "
            f"{C3H4_CORE_SPECIES_FLOOR} (a run that silently dropped its seed "
            f"mechanism is not a valid run)")
    if seed_rxn_count <= 0:
        reasons.append("no family=='seed' reactions in the written artifact "
                       "(seed mechanism not actually loaded)")
    if out["output_tree"]["status"] != "PASS":
        reasons.append("output tree missing/invalid")
    if out["physical_validity"]["status"] != "PASS":
        reasons.append("final profile non-physical (see physical_validity)")
    if reasons:
        out["status"] = "FAIL"
        out["reason"] = "; ".join(reasons)
    else:
        out["status"] = "PASS"

    # Parity vs RMG-Py baseline: RECORDED FINDING, not hard.
    base_path = os.path.join(BASE, "c3h4", "summary.json")
    parity = {}
    if os.path.exists(base_path):
        base = json.load(open(base_path))
        lab2c, rmgpu_core_sp = load_core_artifact(core_yaml_path)
        rxnjson = json.load(open(os.path.join(run_root, "reactions",
                                              "reactions.json")))
        rmgpu_core_rx = _reaction_keys_from_json(rxnjson, lab2c)
        rmg_core_sp = set(base.get("core_species", []))
        rmg_core_rx = set(base.get("core_reactions_keys", []))
        parity = {
            "core_species": setdiff_report("core_species", rmgpu_core_sp, rmg_core_sp),
            "core_reactions": setdiff_report("core_reactions", rmgpu_core_rx, rmg_core_rx),
            "note": "c3h4 parity is expected to diverge (different screening "
                    "criterion + seed handling); recorded finding, not a gate.",
        }
    out["parity_vs_rmgpy"] = parity
    return out


# ---------------------------------------------------------------------------
# Check 5: output tree (both examples)
# ---------------------------------------------------------------------------
def check_output_tree(run_root=RUN_SUPERMIN, label=None):
    out = {"status": "pending", "missing": [], "invalid": []}
    import yaml
    # existence
    for rel in PLAN123_FILES:
        p = os.path.join(run_root, rel)
        if not os.path.exists(p):
            out["missing"].append(rel)
    # validity
    try:
        art = yaml.safe_load(open(os.path.join(run_root, "mechanism", "core.yaml")))
        assert "core" in art and "species" in art["core"] and "reactions" in art["core"]
    except Exception as e:
        out["invalid"].append("mechanism/core.yaml: %r" % (e,))
    try:
        meta = yaml.safe_load(open(os.path.join(run_root, "profiles", "reactor",
                                                "metadata.yaml")))
        assert "units" in meta and "reactor" in meta
    except Exception as e:
        out["invalid"].append("profiles/reactor/metadata.yaml: %r" % (e,))
    # core.yaml loads back via the schema
    try:
        from rmgpu.schemas.mechanism import load_mechanism
        lm = load_mechanism(os.path.join(run_root, "mechanism", "core.yaml"))
        out["core_yaml_loads_back"] = True
        out["core_yaml_species"] = len(lm.core.species)
        out["core_yaml_reactions"] = len(lm.core.reactions)
    except Exception as e:
        out["core_yaml_loads_back"] = False
        out["invalid"].append("core.yaml schema load: %r" % (e,))
    # time_series.csv columns
    try:
        with open(os.path.join(run_root, "profiles", "reactor", "time_series.csv")) as f:
            rows = list(csv.reader(f))
        header = rows[0]
        out["profile_rows"] = len(rows) - 1
        out["profile_cols"] = len(header)
        out["profile_header_is_time_first"] = header[0].startswith("time")
        # Re-simulation of the final core (RECORDED, not hard - see module note).
        try:
            resim = _resimulate_final(run_root)
            out["resimulate_final"] = resim
        except Exception as e:
            out["resimulate_final"] = {"ok": False, "reason": repr(e)}
    except Exception as e:
        out["invalid"].append("time_series.csv: %r" % (e,))
    ok = (not out["missing"]) and (not out["invalid"]) and out.get("core_yaml_loads_back", False)
    out["status"] = "PASS" if ok else "FAIL"
    return out


def _resimulate_final(root):
    """Load core.yaml, re-run the loop's simulate() once on the final core
    mechanism, and compare the resulting final mole-fraction row to the last
    row of profiles/reactor/time_series.csv. RECORDED, NOT A HARD CHECK: this
    re-simulates the full final core from the INITIAL state over a fresh
    5/char timescale - a different, longer-time problem than the loop's final
    snapshot simulation (which started from the previous iteration's state),
    so a large diff is expected and does not invalidate the run. The physical
    validity of the WRITTEN profile is the hard check (check_physical_validity).
    Returns max abs diff + within-tol (tol 1e-3, loose)."""
    import torch  # noqa: F401
    from rmgpu.core.model import Species, Reaction  # noqa: F401
    from rmgpu.molecule.molecule import Molecule
    from rmgpu.reactor.simulator import (RateParam, simulate_mole_fractions,
                                         characteristic_rate)
    import yaml
    art = yaml.safe_load(open(os.path.join(root, "mechanism", "core.yaml")))
    core = art["core"]
    sp_map = {}
    for sp in core["species"]:
        try:
            sp_map[sp["label"]] = Molecule(smiles=sp["smiles"])
        except Exception:
            pass
    rps = []
    for r in core["reactions"]:
        params = r["rate"]["params"]
        rp = RateParam(A=params.get("A", 0.0), n=params.get("n", 0.0),
                       Ea=params.get("Ea", 0.0), T0=params.get("T0", 1.0),
                       dS=params.get("dS", 0.0), dH=params.get("dH", 0.0),
                       family=r.get("family"), source=r.get("source", "ml"))
        react = [sp_map[x] for x in r["reactants"] if x in sp_map]
        prod = [sp_map[x] for x in r["products"] if x in sp_map]
        if len(react) != len(r["reactants"]) or len(prod) != len(r["products"]):
            continue
        rps.append((rp, r["reactants"], r["products"]))
    if not rps:
        return {"ok": False, "reason": "no reconstructable reactions"}
    from rmgpu.units import Quantity
    imf = {}
    T = 1000.0
    P = 1.0e5
    try:
        run_doc = yaml.safe_load(open(os.path.join(root, "run.yaml")))
        T = float(Quantity(run_doc["reactors"][0]["temperature"]["value"],
                           run_doc["reactors"][0]["temperature"]["unit"]).to_si())
        P = float(Quantity(run_doc["reactors"][0]["pressure"]["value"],
                           run_doc["reactors"][0]["pressure"]["unit"]).to_si())
        imf = run_doc["reactors"][0]["initial_mole_fractions"] or {}
    except Exception:
        imf = {}
    with open(os.path.join(root, "profiles", "reactor", "time_series.csv")) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    final_written = [float(x) for x in rows[-1][1:]]
    written_species_cols = [h.replace("_molefrac", "") for h in header[1:]]
    keys = []
    seen = set()
    for rp, rk, pk in rps:
        for k in rk + pk:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    keys.sort()
    idx = {k: i for i, k in enumerate(keys)}
    nu = [[0.0] * len(keys) for _ in rps]
    rp_list = []
    for j, (rp, rk, pk) in enumerate(rps):
        rp_list.append(rp)
        for k in rk:
            nu[j][idx[k]] -= 1
        for k in pk:
            nu[j][idx[k]] += 1
    init = [0.0] * len(keys)
    for i, k in enumerate(keys):
        init[i] = float(imf.get(k, 0.0))
    char = characteristic_rate(nu, rp_list, T, P, init)
    t_end = 5.0 / char if char > 0 else 1.0
    t_end = max(1e-9, min(t_end, 1e12))
    try:
        prof = simulate_mole_fractions(keys, nu, rp_list, T, P, init, t_end,
                                       h=max(t_end / 64, 1e-18))
    except Exception as e:
        return {"ok": False, "reason": "re-simulation raised %r" % (e,)}
    from rdkit import Chem

    def ckey(sm):
        m = Chem.MolFromSmiles(sm)
        if m is None:
            return sm
        try:
            Chem.SanitizeMol(m)
        except Exception:
            pass
        return Chem.MolToSmiles(m)
    label_to_canon = {sp["label"]: canon_smiles(sp["smiles"])
                      for sp in core["species"]}
    sim_canon = {label_to_canon.get(k, ckey(k)): float(v)
                 for k, v in zip(keys, prof.ys[-1])}
    written_canon = {ckey(c): v for c, v in zip(written_species_cols, final_written)}
    common = [k for k in sim_canon if k in written_canon]
    if not common:
        return {"ok": False, "reason": "no overlapping species to compare"}
    diffs = [abs(sim_canon[k] - written_canon[k]) for k in common]
    maxdiff = max(diffs)
    return {"ok": True, "compared_species": len(common),
            "max_abs_diff": float(maxdiff),
            "within_tol": bool(maxdiff < 1e-3),
            "note": "re-simulation of the final core from the initial state; "
                    "RECORDED, not a hard check (different time problem than "
                    "the loop's final snapshot sim)."}


# ---------------------------------------------------------------------------
# Check 6: provenance (both examples)
# ---------------------------------------------------------------------------
def check_provenance(run_root=RUN_SUPERMIN):
    import yaml
    out = {"status": "pending"}
    p = os.path.join(run_root, "provenance.yaml")
    if not os.path.exists(p):
        out["status"] = "FAIL"
        out["reason"] = "provenance.yaml missing"
        return out
    prov = yaml.safe_load(open(p))
    git = str(prov.get("rmgpu_git", ""))
    rmgdb = str(prov.get("rmgdb_hash", ""))
    ml = prov.get("ml_checkpoint_hashes", {}) or {}
    versions = prov.get("versions", {}) or {}
    ok_git = len(git) >= 7 and not git.startswith("unknown")
    ok_db = len(rmgdb) >= 8 and not rmgdb.startswith("unknown")
    ok_ml = any(len(str(v)) >= 16 for v in ml.values())
    ok_ver = any(str(v) not in ("", "unknown") for v in versions.values())
    out.update({
        "rmgpu_git": git, "rmgdb_hash": rmgdb,
        "ml_checkpoint_hashes": {k: str(v)[:12] + "..." for k, v in ml.items()},
        "versions": versions,
        "ok_git": ok_git, "ok_db": ok_db, "ok_ml": ok_ml, "ok_versions": ok_ver,
        "seed_mechanisms": prov.get("seed_mechanisms", []),
    })
    out["status"] = "PASS" if (ok_git and ok_db and ok_ml and ok_ver) else "FAIL"
    return out


# ---------------------------------------------------------------------------
def main():
    results = {"job": "06", "step": "07-gate"}
    results["subgate"] = check_subgate()
    results["superminimal_run_and_parity"] = check_superminimal()
    results["superminimal_divergence_cause"] = _divergence_cause()
    results["c3h4"] = check_c3h4()
    results["output_tree"] = {
        "superminimal": check_output_tree(RUN_SUPERMIN, "superminimal"),
        "c3h4": results["c3h4"].get("output_tree"),
    }
    results["physical_validity"] = {
        "superminimal": check_physical_validity(RUN_SUPERMIN),
        "c3h4": results["c3h4"].get("physical_validity"),
    }
    results["provenance"] = {
        "superminimal": check_provenance(RUN_SUPERMIN),
        "c3h4": check_provenance(RUN_C3H4),
    }

    # HONEST hard checks (job-06/step-07): the gate cannot lie. A run that
    # "completes" by silently dropping its seed mechanism, a non-physical
    # profile, a missing/invalid output tree, fake coverage, or fake provenance
    # all FAIL.
    sm = results["superminimal_run_and_parity"]
    c3 = results["c3h4"]
    hard = {
        "subgate": results["subgate"]["status"] == "PASS",
        "run_completes_superminimal": sm["status"] in ("PASS", "MAX_ITER"),
        "c3h4": c3["status"] == "PASS",
        "output_tree_superminimal":
            results["output_tree"]["superminimal"]["status"] == "PASS",
        "output_tree_c3h4":
            (results["output_tree"]["c3h4"] or {}).get("status") == "PASS",
        "physical_validity_superminimal":
            results["physical_validity"]["superminimal"]["status"] == "PASS",
        "physical_validity_c3h4":
            (results["physical_validity"]["c3h4"] or {}).get("status") == "PASS",
        "provenance_superminimal":
            results["provenance"]["superminimal"]["status"] == "PASS",
        "provenance_c3h4": results["provenance"]["c3h4"]["status"] == "PASS",
    }
    results["hard_checks"] = hard
    results["all_hard_pass"] = all(hard.values())

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1, default=str)

    # console summary
    print("=== job-06 gate (honest, step-07) ===")
    print("sub-gate: %s (VdP %.2e, Lindemann %.2e)" %
          (results["subgate"]["status"],
           results["subgate"]["van_der_pol_mu10_max_abs_diff"],
           results["subgate"]["lindemann_toy_max_abs_diff"]))
    print("superminimal run: %s, iterations=%s, core %d spc/%d rxn, edge %d spc/%d rxn, %.0fs"
          % (sm["status"], sm["iterations"], sm["core_species_count"],
             sm["core_reaction_count"], sm["edge_species_count"],
             sm["edge_reaction_count"], sm["elapsed_seconds"]))
    print("  coverage: est=%s gaps=%s"
          % (sm.get("estimation_counts"), sm.get("coverage_gaps")))
    par = sm.get("parity_vs_rmgpy", {})
    if par:
        cs = par["core_species"]
        cr = par["core_reactions"]
        print("  core_species parity: %s (shared %d, rmgpu-only %d, ref-only %d)"
              % ("IDENTICAL" if cs["identical"] else "DIVERGENT",
                 len(cs["shared"]), len(cs["only_in_rmgpu"]), len(cs["only_in_reference"])))
        print("  core_reactions parity: %s (shared %d, rmgpu-only %d, ref-only %d)"
              % ("IDENTICAL" if cr["identical"] else "DIVERGENT",
                 len(cr["shared"]), len(cr["only_in_rmgpu"]), len(cr["only_in_reference"])))
    pv = results["physical_validity"]
    print("physical validity: superminimal=%s c3h4=%s"
          % (pv["superminimal"]["status"], pv["c3h4"]["status"]))
    print("c3h4: %s (core %s spc, seed rxn in artifact %s, seed floor %s)"
          % (c3["status"], c3.get("core_species_count"),
             c3.get("seed_reaction_count_in_artifact"),
             c3.get("core_species_floor")))
    if c3.get("reason"):
        print("  c3h4 reason: %s" % c3["reason"])
    print("output_tree: superminimal=%s c3h4=%s"
          % (results["output_tree"]["superminimal"]["status"],
             (results["output_tree"]["c3h4"] or {}).get("status")))
    print("provenance: superminimal=%s c3h4=%s"
          % (results["provenance"]["superminimal"]["status"],
             results["provenance"]["c3h4"]["status"]))
    print("HARD CHECKS:")
    for k, v in hard.items():
        print("  %-32s %s" % (k, "PASS" if v else "FAIL"))
    print("=> ALL PASS" if results["all_hard_pass"] else "=> FAIL")
    sys.exit(0 if results["all_hard_pass"] else 1)


if __name__ == "__main__":
    main()
