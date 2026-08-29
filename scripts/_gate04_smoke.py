"""Throwaway smoke: verify resolver wiring + DB label lookups for the gate.

Run: /home/jackson/miniforge3/envs/rmgpu/bin/python scripts/_gate04_smoke.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rmgpu.db import Databases  # noqa: E402
from rmgpu.data.estimation import EstimationCounts, estimate_thermo, estimate_kinetics  # noqa: E402
from rmgpu.ml.thermo_estimator import ThermoML, MLCoverageError  # noqa: E402
from rmgpu.ml.kinetics_estimator import KineticsML  # noqa: E402

ref = json.loads((REPO / "gates/baselines/thesis_test/thermo.json").read_text())
rxn = json.loads((REPO / "gates/baselines/thesis_test/reactions.json").read_text())

class Estimators:
    def __init__(self):
        self.kinetics = KineticsML()
        self.thermo = self.kinetics.thermo_ml

print("[smoke] loading estimators", flush=True)
est = Estimators()
db = Databases.from_config({"thermo_libraries": ["primaryThermoLibrary"],
                            "reaction_libraries": ["primaryH2O2"]})
print("[smoke] db kinetics libs:", db.kinetics.get_library_names(), flush=True)
print("[smoke] primaryH2O2 rxn count:",
      db.kinetics.get_library_reaction_count("primaryH2O2"), flush=True)

# 1) which of the 46 labels exist in primaryThermoLibrary?
hits = 0
miss = []
for smi, v in ref["species"].items():
    e = db.thermo.get_entry_grouped_by_label(v["label"], "primaryThermoLibrary")
    if e is not None:
        hits += 1
    else:
        miss.append((smi, v["label"]))
print(f"[smoke] PTL label hits: {hits}/{len(ref['species'])}; miss={miss}", flush=True)

# 2) estimate_thermo on a few: a PTL species, and the two non-PTL (CC, C#C)
for smi in ["CCC", "[H][H]", "CC", "C#C"]:
    v = ref["species"][smi]
    c = EstimationCounts()
    try:
        p = estimate_thermo({"label": v["label"], "smiles": smi}, db.thermo, est,
                            counts=c, libraries=["primaryThermoLibrary"])
        print(f"[smoke] thermo {smi} {v['label']}: Hf298={p.Hf298:.1f} S298={p.S298:.2f} "
              f"Cp300={p.Cp_model.get_heat_capacity(300.0):.2f} "
              f"src={p.uncertainties.get('source')} counts={c.as_dict()}")
    except MLCoverageError as e:
        print(f"[smoke] thermo {smi} {v['label']}: COVERAGE-ERROR {e} counts={c.as_dict()}")

# 3) kinetics: a couple of GRI reactions through the resolver + a raw ML probe
def mk_rxn(smi):
    rec = rxn["reactions"][smi]
    return {
        "label": smi[:60],
        "reactants": [{"label": f"r{j}", "smiles": s} for j, s in enumerate(rec["reactants"])],
        "products": [{"label": f"p{j}", "smiles": s} for j, s in enumerate(rec["products"])],
        "reaction_smiles": smi,
    }

import numpy as np
R = 8.314472
for smi in list(rxn["reactions"].keys())[:6]:
    rec = rxn["reactions"][smi]
    c = EstimationCounts()
    try:
        model, deg = estimate_kinetics(mk_rxn(smi), db.kinetics, est, counts=c,
                                       libraries=["primaryH2O2"], degeneracy=1.0)
        A = float(model.A); n = float(model.n); Ea = float(model.Ea)
        # raw k from the stored model (SI if library, CGS if ML)
        k300_model = A * (300.0/1.0)**n * np.exp(-Ea/(R*300.0))
        k300_ref = rec["k300"]
        molec = rec["molecularity"]
        k300_si = k300_model * 1e-6*(molec-1)
        print(f"[smoke] kin {smi[:48]} bucket={rec['bucket']} "
              f"A={A:.3e} n={n:.3f} Ea={Ea:.1f} deg={deg} "
              f"k300_model={k300_model:.3e} k300_SI={k300_si:.3e} k300_ref={k300_ref:.3e} "
              f"ratio_SI/ref={k300_si/k300_ref:.3f} counts={c.as_dict()}")
    except MLCoverageError as e:
        print(f"[smoke] kin {smi[:48]}: COVERAGE-ERROR {e} counts={c.as_dict()}")

print("[smoke] done")
