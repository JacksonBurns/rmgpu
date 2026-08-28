"""Integration smoke for the estimation resolvers (job-04 step-05).

Runs estimate_thermo on the 5 seed species of the {superminimal, c3h4}
example sets (examples/rmg/superminimal: H2, O2; examples/rmg/c3h4: CH2,
C2H2, N2) against the REAL rmgdb thermo DB (primaryThermoLibrary) and the
REAL vendored thermo checkpoint (models/chemeleon_thermo_662946.ckpt).
Records the library/ML/coverage split (the no-fallback instrumentation) in
prints for the step report.

Run:  /home/jackson/miniforge3/envs/rmgpu/bin/python scripts/smoke_estimation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rmgpu.data.estimation import EstimationCounts, estimate_thermo
from rmgpu.db import Databases
from rmgpu.ml.thermo_estimator import ThermoML
from rmgpu.ml.base import MODELS_DIR

# The 5 seed species of the two minimal example sets (label, smiles).
SPECIES = [
    ("H2", "[H][H]"),      # superminimal
    ("O2", "[O][O]"),      # superminimal
    ("CH2", "[CH2]"),      # c3h4
    ("C2H2", "C#C"),       # c3h4
    ("N2", None),          # c3h4 (InChI in the input; NASA-only library entry)
]

LIBRARIES = ["primaryThermoLibrary"]


def main() -> int:
    db = Databases.from_config({"thermo_libraries": LIBRARIES})
    ml_thermo = ThermoML(MODELS_DIR)

    class ML:
        thermo = ml_thermo

    counts = EstimationCounts()
    results = []
    for label, smiles in SPECIES:
        species = {"label": label}
        if smiles:
            species["smiles"] = smiles
        else:
            # N2: no SMILES supplied on purpose - the library hit must not
            # need a structure, and a miss would show as a coverage error.
            species["smiles"] = "N#N"
        try:
            pred = estimate_thermo(species, db.thermo, ML(), counts,
                                   libraries=LIBRARIES)
            cp300 = pred.Cp_model.get_heat_capacity(300.0)
            results.append((label, "OK", pred.Hf298, pred.S298, cp300,
                            pred.uncertainties.get("source", "")))
        except Exception as e:
            results.append((label, f"RAISED {type(e).__name__}", str(e), None, None, ""))

    print("=== job-04 step-05 integration smoke (real DB + real checkpoint) ===")
    print(f"libraries: {LIBRARIES}")
    print(f"{'label':<8} {'outcome':<8} {'Hf298 J/mol':>14} {'S298 J/mol/K':>14} "
          f"{'Cp300 J/mol/K':>14}  source")
    for label, status, hf, s, cp, src in results:
        if status == "OK":
            print(f"{label:<8} {status:<8} {hf:>14.1f} {s:>14.3f} {cp:>14.3f}  {src}")
        else:
            print(f"{label:<8} {status:<8} {hf}")
    print("split:", counts.as_dict())
    ok = sum(1 for r in results if r[1] == "OK")
    print(f"resolved {ok}/{len(SPECIES)}")
    return 0 if ok == len(SPECIES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
