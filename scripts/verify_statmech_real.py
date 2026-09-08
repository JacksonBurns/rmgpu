#!/usr/bin/env python3
"""Real sub-gate 1: reconstruct each reference species' conformer from the
dumped mode spec and compare Cp(T) + DoS against RMG-Py's recorded values."""
import sys
import json
import numpy as np

sys.path.insert(0, "/home/jackson/rmgpu/rmgpu")
from rmgpu.statmech.modes import conformer_from_spec

B = "/home/jackson/rmgpu/rmgpu/gates/baselines/propane_branching"


def main():
    d = json.load(open(B + "/statmech.json"))
    T = np.array([300, 500, 700, 1000, 1500], dtype=float)
    # DoS energy grid: use the same the reference used if present, else a
    # representative grid. The reference stores DoS on its own e_list; we
    # compare on the recorded grid (per species).
    worst_dos, worst_cp = 0.0, 0.0
    n_checked = 0
    n_nomodes = 0
    rows = []
    for s in d["species"]:
        label = s["label"]
        modes = s.get("modes", [])
        if not modes:
            n_nomodes += 1
            rows.append((label, None, None, "no-modes"))
            continue
        spec = {
            "E0": s.get("E0", 0.0),
            "spin_multiplicity": s.get("spin_multiplicity", 1),
            "optical_isomers": s.get("optical_isomers", 1),
            "modes": modes,
        }
        c = conformer_from_spec(spec)
        # Cp on the reference's T grid (recorded Cp_T.values)
        cp_rel = None
        cprec = s.get("Cp_T")
        if cprec is not None:
            ref_cp = np.asarray(cprec["values"], dtype=float)
            my_cp = np.array([c.get_heat_capacity(t) for t in cprec["T"]])
            cp_rel = float(np.max(np.abs(my_cp - ref_cp) / np.maximum(np.abs(ref_cp), 1e-30)))
        # DoS on the recorded grid (the reference dumps the key "DOS")
        dos_rel = None
        dos = s.get("DOS")
        if dos is not None and "values" in dos:
            e = np.asarray(dos["e_above"], dtype=float)
            ref_d = np.asarray(dos["values"], dtype=float)
            my_d = np.asarray(c.get_density_of_states(e), dtype=float)
            nz = ref_d > 0
            if nz.any():
                dos_rel = float(np.max(np.abs(my_d[nz] - ref_d[nz]) / np.maximum(ref_d[nz], 1e-30)))
        n_checked += 1
        worst_cp = max(worst_cp, cp_rel) if cp_rel is not None else worst_cp
        worst_dos = max(worst_dos, dos_rel) if dos_rel is not None else worst_dos
        rows.append((label, cp_rel, dos_rel, "ok"))
    for label, cp_rel, dos_rel, tag in rows:
        if tag == "no-modes":
            print(f"  - {label:12s} (no conformer modes - library/NASA species)")
            continue
        cp_s = "n/a" if cp_rel is None else f"{cp_rel:.2e}"
        dos_s = "n/a" if dos_rel is None else f"{dos_rel:.2e}"
        print(f"  - {label:12s} Cp={cp_s}  DoS={dos_s}")
    print()
    print(f"species with modes: {n_checked}   no-modes: {n_nomodes}")
    print(f"worst Cp  rel: {worst_cp:.3e}")
    print(f"worst DoS rel: {worst_dos:.3e}")
    ok = worst_cp < 1e-6 and worst_dos < 1e-6
    print("PASS (1e-6)" if ok else "FAIL (1e-6)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
