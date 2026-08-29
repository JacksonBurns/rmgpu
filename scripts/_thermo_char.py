"""Characterize the thermo Hf298/S298 ML-vs-ref relationship per species.

Run in rmgpu env. Reconstructs the model's Hf298/S298 for all 46 thesis species
and reports the relationship to the RMG-Py reference (is it a scale/offset, a
genuine mismatch, or sign-driven?). Also dumps per-species for the report.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
from rmgpu.ml.kinetics_estimator import KineticsML

ref = json.loads((REPO / "gates/baselines/thesis_test/thermo.json").read_text())
ml = KineticsML().thermo_ml

rows = []
for smi, v in ref["species"].items():
    if "error" in v:
        continue
    p = ml.predict(smi)
    rows.append({
        "smiles": smi, "label": v["label"], "class": v["class"],
        "ref_Hf298_kJ": v["Hf298"] / 1000.0,
        "ml_Hf298_kJ": p.Hf298 / 1000.0,
        "dHf_kJ": (p.Hf298 - v["Hf298"]) / 1000.0,
        "ref_S298": v["S298"],
        "ml_S298": p.S298,
        "dS": p.S298 - v["S298"],
        "ref_Cp300": v["Cp300"], "ml_Cp300": p.Cp_model.get_heat_capacity(300.0),
        "ref_Cp600": v["Cp600"], "ml_Cp600": p.Cp_model.get_heat_capacity(600.0),
        "ref_Cp1000": v["Cp1000"], "ml_Cp1000": p.Cp_model.get_heat_capacity(1000.0),
    })

ref_hf = np.array([r["ref_Hf298_kJ"] for r in rows])
ml_hf = np.array([r["ml_Hf298_kJ"] for r in rows])
print(f"n={len(rows)}")
print(f"ref_Hf298 kJ/mol: min={ref_hf.min():.1f} max={ref_hf.max():.1f} "
      f"mean={ref_hf.mean():.1f}")
print(f"ml_Hf298  kJ/mol: min={ml_hf.min():.1f} max={ml_hf.max():.1f} "
      f"mean={ml_hf.mean():.1f}  (all positive: {(ml_hf>0).all()})")
# Pearson correlation
print(f"pearson corr(ref,ml) Hf298 = {np.corrcoef(ref_hf, ml_hf)[0,1]:.3f}")
# slope/offset of ml = a*ref + b
a, b = np.polyfit(ref_hf, ml_hf, 1)
print(f"linear fit ml = a*ref + b : a={a:.3f} b={b:.1f}")
# For the negative-ref group specifically
neg = [r for r in rows if r["ref_Hf298_kJ"] < 0]
print(f"\nnegative-Hf298 species: {len(neg)}; ml values all in "
      f"[{min(r['ml_Hf298_kJ'] for r in neg):.0f}, "
      f"{max(r['ml_Hf298_kJ'] for r in neg):.0f}] kJ/mol (positive)")

print("\nper-species (label, class, ref_Hf kJ, ml_Hf kJ, dHf kJ, dS J/mol/K):")
for r in rows:
    print(f"  {r['label'][:14]:15} {r['class']:12} ref={r['ref_Hf298_kJ']:8.1f} "
          f"ml={r['ml_Hf298_kJ']:8.1f} dHf={r['dHf_kJ']:+8.1f} dS={r['dS']:+7.2f}")

# S298 and Cp relationships
ref_s = np.array([r["ref_S298"] for r in rows]); ml_s = np.array([r["ml_S298"] for r in rows])
print(f"\nS298: ref mean={ref_s.mean():.1f} ml mean={ml_s.mean():.1f} "
      f"corr={np.corrcoef(ref_s,ml_s)[0,1]:.3f} mean dS={ (ml_s-ref_s).mean():+.2f}")
for T in (300,600,1000):
    rc = np.array([r[f"ref_Cp{int(T)}"] for r in rows]); mc = np.array([r[f"ml_Cp{int(T)}"] for r in rows])
    print(f"Cp({T}): ref mean={rc.mean():.1f} ml mean={mc.mean():.1f} "
          f"mean dCp={(mc-rc).mean():+.2f} median dCp={np.median(mc-rc):+.2f} "
          f"corr={np.corrcoef(rc,mc)[0,1]:.3f}")

out = REPO / "reports" / "thermo_characterization.json"
out.write_text(json.dumps(rows, indent=2))
print(f"\nwrote {out}")
