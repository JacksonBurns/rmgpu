"""Decompose the job-04 thesis-test RED into per-parameter errors.

Two things to establish rigorously for the report:
  (A) Kinetics: is the log10(k) ratio a pure 1/T term (=> Ea bias) with the
      A*T^n term ~ 0? Fit the reference (A,n,Ea) from the 3 SI k(T) points per
      reaction and compare each parameter to the model's.
  (B) Thermo: is the dHf298 error (a) a positive-only-output/domain problem,
      (b) an absolute-scale offset, or (c) genuine in-domain error?

Writes reports/thesis_decomposition.json. Runs the kinetics checkpoint once.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rmgpu.kinetics.models import R  # noqa: E402
from rmgpu.ml.kinetics_estimator import KineticsML  # noqa: E402


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        import numpy as _np
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
        return super().default(o)

RXN = json.loads((REPO / "gates/baselines/thesis_test/reactions.json").read_text())
ref_t = json.loads((REPO / "gates/baselines/thesis_test/thermo.json").read_text())
T_LIST = np.array([300.0, 600.0, 1000.0])


def si_to_cgs_factor(molec: int) -> float:
    return 10.0 ** (6 * (molec - 1))  # m^x/mol^y -> cm^x/mol^y


def fit_arrhenius(k_cgs: np.ndarray, T: np.ndarray):
    """ln k = lnA + n lnT - Ea/(R T). Returns (A, n, Ea)."""
    lnk = np.log(k_cgs)
    X = np.column_stack([np.ones(3), np.log(T), 1.0 / T])
    coef, *_ = np.linalg.lstsq(X, lnk, rcond=None)
    lnA, n, c2 = coef
    return np.exp(lnA), n, -c2 * R


def stats(a):
    a = np.asarray(a, float)
    if a.size == 0:
        return {"N": 0}
    return {"N": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "std": float(a.std()), "p95_abs": float(np.percentile(np.abs(a), 95)),
            "max_abs": float(np.abs(a).max()), "min": float(a.min()), "max": float(a.max())}


print("[decomp] loading kinetics checkpoint...", flush=True)
kin = KineticsML()
reactions = RXN["reactions"]

d_logA, d_n, d_Ea, d_k600 = [], [], [], []
per_rxn = {}
for i, (smi, rec) in enumerate(reactions.items()):
    try:
        pred = kin.predict(smi)  # A in CGS (per-site, PLAN 3b), n, Ea (J/mol)
    except Exception as e:
        continue
    A_ml, n_ml, Ea_ml = float(pred.A), float(pred.n), float(pred.Ea)
    k_si = np.array([rec["k300"], rec["k600"], rec["k1000"]], float)
    if np.any(~np.isfinite(k_si)) or np.any(k_si <= 0):
        continue
    k_cgs = k_si * si_to_cgs_factor(rec["molecularity"])
    A_ref, n_ref, Ea_ref = fit_arrhenius(k_cgs, T_LIST)
    d_logA.append(np.log10(A_ml / A_ref))
    d_n.append(n_ml - n_ref)
    d_Ea.append(Ea_ml - Ea_ref)
    # check 2 at 600K for cross-check vs the gate
    k_ref_600 = k_si[1]
    k_ml_600 = A_ml * 600.0 ** n_ml * np.exp(-Ea_ml / (R * 600.0)) * \
        10.0 ** (-6 * (rec["molecularity"] - 1))
    d_k600.append(np.log10(k_ml_600 / k_ref_600))
    per_rxn[smi] = {"bucket": rec["bucket"], "d_logA": float(np.log10(A_ml / A_ref)),
                    "d_n": float(n_ml - n_ref), "d_Ea_kJ": float((Ea_ml - Ea_ref) / 1000.0),
                    "Ea_ml_kJ": float(Ea_ml / 1000.0), "Ea_ref_kJ": float(Ea_ref / 1000.0)}
    if (i + 1) % 50 == 0:
        print(f"[decomp] {i+1}/{len(reactions)}", flush=True)

print("[decomp] loading thermo checkpoint... (reusing kinetics' thermo_ml)", flush=True)
mlt = kin.thermo_ml
th = []
for smi, v in ref_t["species"].items():
    if "error" in v:
        continue
    p = mlt.predict(smi)
    th.append({"label": v["label"], "class": v["class"],
               "ref_Hf_kJ": v["Hf298"] / 1000.0, "ml_Hf_kJ": p.Hf298 / 1000.0,
               "dHf_kJ": (p.Hf298 - v["Hf298"]) / 1000.0,
               "ref_S298": v["S298"], "ml_S298": p.S298, "dS": p.S298 - v["S298"]})

out = {
    "kinetics_per_parameter": {
        "description": "d_logA = log10(A_ml/A_ref), d_n = n_ml - n_ref, "
                      "d_Ea = Ea_ml - Ea_ref (J/mol); ref fit from 3 SI k(T) points "
                      "(converted to CGS per molecularity). N = reactions with finite ref.",
        "N": len(d_logA),
        "d_log10_A": stats(d_logA),
        "d_n": stats(d_n),
        "d_Ea_J_mol": stats(d_Ea),
        "d_Ea_kJ_mol": {k: (v / 1000.0 if isinstance(v, float) else v)
                        for k, v in stats(d_Ea).items()},
        "log10_k_ratio_600K_crosscheck": stats(d_k600),
    },
    "thermo_per_species": th,
    "thermo_dHf_by_domain": {
        "ref_negative": stats([r["dHf_kJ"] for r in th if r["ref_Hf_kJ"] < 0]),
        "ref_nonnegative": stats([r["dHf_kJ"] for r in th if r["ref_Hf_kJ"] >= 0]),
        "ref_near_zero_abs_lt5": stats([r["dHf_kJ"] for r in th if abs(r["ref_Hf_kJ"]) < 5]),
    },
}
OUT = REPO / "reports" / "thesis_decomposition.json"
OUT.write_text(json.dumps(out, indent=2, cls=NpEncoder))
print(f"\n[decomp] wrote {OUT}")
print(f"kinetics N={len(d_Ea)}  d_logA mean={np.mean(d_logA):+.3f} p95={np.percentile(np.abs(d_logA),95):.3f}")
print(f"          d_n mean={np.mean(d_n):+.3f} p95={np.percentile(np.abs(d_n),95):.3f}")
print(f"          d_Ea mean={np.mean(d_Ea)/1000:+.1f} kJ/mol p95={np.percentile(np.abs(d_Ea),95)/1000:.1f} kJ/mol max={np.abs(d_Ea).max()/1000:.1f}")
print(f"thermo N={len(th)}  dHf ref<0: mean={np.mean([r['dHf_kJ'] for r in th if r['ref_Hf_kJ']<0]):+.1f} kJ/mol")
nz = [r['dHf_kJ'] for r in th if abs(r['ref_Hf_kJ'])<5]
print(f"          dHf |ref|<5 (elements/near-zero): N={len(nz)} mean={np.mean(nz):+.1f} max={np.abs(nz).max():.1f} kJ/mol")
