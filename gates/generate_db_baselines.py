#!/usr/bin/env python
"""Generate the RMG-Py-side baselines for the job-02 gate.

Runs in the ``rmg_env`` conda environment (the only env with a compiled
rmgpy). Loads the same library files the rmgdb build consumed
(/home/jackson/rmgpu/RMG-database/input/{thermo,kinetics}) and records, for
the fixed gate library set:

  counts       entry count per library (RMG-Py's own count)
  content_hash sha256 of the canonical YAML dump of every entry in
               primaryThermoLibrary (thermo) and primaryH2O2 (kinetics),
               normalized with the SHARED gates/normalizer.py so both sides
               of the gate hash identical structures
  lookup       the 25 gate species (fixed seed) with their canonical
               normalized entries, for the lookup-parity check
  rate         k(300 K, 1 bar) and k(1000 K, 1 bar) for every evaluable
               reaction of the three gate kinetics libraries, in per-library
               file order (position == the order rmgdb stored the rows)

Reactions whose rate rmgdb cannot reconstruct from its tables (Chebyshev
coefficients not stored, nested MultiArrhenius inside PDepArrhenius stored as
NULL, nested T0 != 1 not stored, non-finite rates from negative-A blocks,
no rmgpu assembler) are listed in ``excluded`` with a reason code and are NOT
part of the round-trip; they double as the documented rmgdb coverage gaps.

Output: gates/baselines/db_baselines.json

Usage:
    /home/jackson/miniforge3/envs/rmg_env/bin/python gates/generate_db_baselines.py
"""
import hashlib
import json
import logging
import math
import random
import sys
from collections import Counter
from pathlib import Path

# The shared normalizer is pure python + yaml; import it from this directory
# so BOTH sides of the gate hash byte-identical structures.
sys.path.insert(0, str(Path(__file__).parent))
from normalizer import (  # noqa: E402
    normalize_thermo_entry,
    normalize_kinetics_entry,
    _rows_from_rmgpy_thermo,
    _raw_from_rmgpy_kinetics,
    canonical_dump,
)

logging.disable(logging.CRITICAL)

from rmgpy.data.thermo import ThermoDatabase  # noqa: E402
from rmgpy.data.kinetics import KineticsDatabase  # noqa: E402
from rmgpy.kinetics import Arrhenius  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed gate library set (the job-02 gate compares these, nothing else)
# ---------------------------------------------------------------------------
THERMO_LIBRARIES = ["primaryThermoLibrary", "BurcatNS", "BurkeH2O2", "NOx2018"]
KINETICS_LIBRARIES = ["primaryH2O2", "primaryNitrogenLibrary", "NOx2018"]
CONTENT_HASH = {"thermo": "primaryThermoLibrary", "kinetics": "primaryH2O2"}
LOOKUP_LIBRARY = "primaryThermoLibrary"
LOOKUP_N = 25
LOOKUP_SEED = 42
T1, T2, P = 300.0, 1000.0, 1e5  # K, K, Pa

THERMO_DB_PATH = "/home/jackson/rmgpu/RMG-database/input/thermo"
KINETICS_DB_PATH = "/home/jackson/rmgpu/RMG-database/input/kinetics"
OUT = Path(__file__).parent / "baselines" / "db_baselines.json"

# Model types rmgpu.data.kinetics.assemble_rate_model can reconstruct.
ASSEMBLABLE = {
    "Arrhenius", "MultiArrhenius", "Lindemann", "Troe", "ThirdBody",
    "PDepArrhenius", "MultiPDepArrhenius",
}


def exclusion_reason(d) -> str | None:
    """Reason a reaction's rate cannot be reconstructed from rmgdb tables.

    Returns None when rmgpu can build the model and it must round-trip.
    These reasons ARE the documented rmgdb coverage gaps (check 5).
    """
    m = type(d).__name__
    if m == "Chebyshev":
        return "chebyshev: coefficients not stored by rmgdb"
    if m not in ASSEMBLABLE:
        return f"{m}: no rmgpu assembler"
    if m in ("Troe", "Lindemann", "ThirdBody"):
        nested = [a for a in (getattr(d, "arrheniusHigh", None),
                              getattr(d, "arrheniusLow", None)) if a is not None]
        for a in nested:
            t0 = a.T0
            t0v = float(t0.value) if t0 is not None else 1.0
            if t0v != 1.0:
                return "falloff: nested T0 != 1 not stored by rmgdb"
    if m == "PDepArrhenius":
        for a in d.arrhenius:
            if not isinstance(a, Arrhenius):
                return (f"PDepArrhenius: nested {type(a).__name__} at a "
                        "pressure point stored as NULL by rmgdb")
    if m == "MultiPDepArrhenius":
        for b in d.arrhenius:
            for a in b.arrhenius:
                if not isinstance(a, Arrhenius):
                    return (f"MultiPDepArrhenius: nested {type(a).__name__} "
                            "stored as NULL by rmgdb")
    return None


def main() -> int:
    print("Loading RMG-Py thermo database ...")
    tdb = ThermoDatabase()
    tdb.load(THERMO_DB_PATH)
    print("Loading RMG-Py kinetics database (this takes a few minutes) ...")
    kdb = KineticsDatabase()
    kdb.load(KINETICS_DB_PATH, families=[], depositories=[])

    # ---------------- check 1: counts ----------------
    counts = {"thermo": {}, "kinetics": {}}
    for lib in THERMO_LIBRARIES:
        counts["thermo"][lib] = len(tdb.libraries[lib].entries)
        print(f"  thermo   {lib:28} {counts['thermo'][lib]:6d}")
    for lib in KINETICS_LIBRARIES:
        counts["kinetics"][lib] = len(kdb.libraries[lib].entries)
        print(f"  kinetics {lib:28} {counts['kinetics'][lib]:6d}")

    # ---------------- check 2: content hashes ----------------
    content_hash = {"thermo": {}, "kinetics": {}}

    tlib = tdb.libraries[CONTENT_HASH["thermo"]]
    entries = [normalize_thermo_entry(e.label, _rows_from_rmgpy_thermo(e))
               for e in sorted(tlib.entries.values(), key=lambda e: e.label)]
    text = canonical_dump(entries)
    content_hash["thermo"][CONTENT_HASH["thermo"]] = hashlib.sha256(
        text.encode()).hexdigest()
    print(f"  content hash thermo    {CONTENT_HASH['thermo']}: "
          f"{content_hash['thermo'][CONTENT_HASH['thermo']][:16]}...")

    klib = kdb.libraries[CONTENT_HASH["kinetics"]]
    entries = [normalize_kinetics_entry(e.label, _raw_from_rmgpy_kinetics(e))
               for e in klib.entries.values()]
    entries.sort(key=lambda e: e["label"])
    text = canonical_dump(entries)
    content_hash["kinetics"][CONTENT_HASH["kinetics"]] = hashlib.sha256(
        text.encode()).hexdigest()
    print(f"  content hash kinetics  {CONTENT_HASH['kinetics']}: "
          f"{content_hash['kinetics'][CONTENT_HASH['kinetics']][:16]}...")

    # ---------------- check 3: lookup set ----------------
    random.seed(LOOKUP_SEED)
    labels = list(tdb.libraries[LOOKUP_LIBRARY].entries.keys())
    picked = random.sample(labels, min(LOOKUP_N, len(labels)))
    lookup = []
    for lab in picked:
        e = tdb.libraries[LOOKUP_LIBRARY].entries[lab]
        lookup.append({
            "label": lab,
            "entry": normalize_thermo_entry(lab, _rows_from_rmgpy_thermo(e)),
        })
    print(f"  lookup species ({len(lookup)}): {picked}")

    # ---------------- check 4: rate round-trip values ----------------
    rate = []
    excluded = []
    for lib in KINETICS_LIBRARIES:
        entries_list = list(kdb.libraries[lib].entries.values())
        for pos, e in enumerate(entries_list):
            d = e.data
            m = type(d).__name__
            reason = exclusion_reason(d)
            k300 = k1000 = None
            if reason is None:
                try:
                    k300 = d.get_rate_coefficient(T1, P)
                    k1000 = d.get_rate_coefficient(T2, P)
                except Exception as ex:  # RMG-Py itself cannot evaluate it
                    reason = f"{m}: RMG-Py evaluation failed ({ex!r})"
                    k300 = k1000 = None
                else:
                    if not (math.isfinite(k300) and math.isfinite(k1000)):
                        reason = (f"{m}: non-finite RMG-Py rate "
                                  "(negative-A block in source)")
                        k300 = k1000 = None
            if reason is not None:
                excluded.append({"library": lib, "label": e.label,
                                 "index": e.index, "model": m, "reason": reason})
                continue
            rate.append({"library": lib, "pos": pos, "label": e.label,
                         "index": e.index, "model": m,
                         "k300_1e5": k300, "k1000_1e5": k1000})
    print(f"  rate: {len(rate)} evaluable, {len(excluded)} excluded")
    for r, c in Counter(x["reason"].split(":")[0] for x in excluded).items():
        print(f"    excluded[{r}]: {c}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "T": [T1, T2, P],
        "counts": counts,
        "content_hash": content_hash,
        "lookup": lookup,
        "rate": rate,
        "excluded": excluded,
    }, indent=1))
    print(f"Saved baselines to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
