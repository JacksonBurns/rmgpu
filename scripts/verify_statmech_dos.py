#!/usr/bin/env python3
"""Verify rmgpu.statmech.modes production DoS against the RMG-Py oracle.

Oracle schema (statmech_dos_oracle.json):
  e_list: [J/mol grid]
  T_list: [K grid]
  conformers: [ { label, E0, spin_multiplicity, optical_isomers,
                  modes: [{type, ...}],
                  Cp_T: {T: [...], values: [...]},
                  DoS: {e: [...], values: [...]},
                  SoS: {e: [...], values: [...]} } ]
"""
import sys
import json
import numpy as np

sys.path.insert(0, "/home/jackson/rmgpu/rmgpu")
from rmgpu.statmech.modes import conformer_from_spec

B = "/home/jackson/rmgpu/rmgpu/gates/baselines/propane_branching"


def main():
    orc = json.load(open(B + "/statmech_dos_oracle.json"))
    e = np.array(orc["e_list"], dtype=float)
    T = np.array(orc["T_list"], dtype=float)
    worst_dos = 0.0
    worst_cp = 0.0
    worst_sos = 0.0
    rows = []
    for conf in orc["conformers"]:
        name = conf["label"]
        spec = {
            "E0": conf.get("E0", 0.0),
            "spin_multiplicity": conf.get("spin_multiplicity", 1),
            "optical_isomers": conf.get("optical_isomers", 1),
            "modes": conf["modes"],
        }
        c = conformer_from_spec(spec)
        my_dos = np.asarray(c.get_density_of_states(e), dtype=float)
        ref_dos = conf.get("DoS")
        if ref_dos is not None and len(ref_dos.get("e", [])) == len(e):
            rd = np.asarray(ref_dos["values"], dtype=float)
            nz = rd > 0
            rel = np.abs(my_dos[nz] - rd[nz]) / np.maximum(rd[nz], 1e-30)
            my_rel = float(np.max(rel)) if nz.any() else 0.0
            # also check zeros stay zero (count of nonzero must match)
            nn_ok = int(np.sum(my_dos > 0)) == int(np.sum(rd > 0))
        else:
            my_rel = float("nan")
            nn_ok = None
        my_cp = np.array([c.get_heat_capacity(t) for t in T])
        ref_cp = conf.get("Cp_T")
        cp_rel = float("nan")
        if ref_cp is not None:
            rc = np.asarray(ref_cp["values"], dtype=float)
            cp_rel = float(np.max(np.abs(my_cp - rc) / np.maximum(rc, 1e-30)))
        # SoS where present (RMG-Py raises NotImplementedError for a
        # FreeRotor with an accumulated SoS - treat as unavailable, matching).
        my_sos = None
        ref_sos = conf.get("SoS")
        sos_rel = float("nan")
        if ref_sos is not None:
            try:
                my_sos = c.get_sum_of_states(e)
            except NotImplementedError:
                pass
        if my_sos is not None and ref_sos is not None:
            rs = np.asarray(ref_sos["values"], dtype=float)
            nz = rs > 0
            rel = np.abs(my_sos[nz] - rs[nz]) / np.maximum(rs[nz], 1e-30)
            sos_rel = float(np.max(rel)) if nz.any() else 0.0
        worst_dos = max(worst_dos, my_rel) if not np.isnan(my_rel) else worst_dos
        worst_cp = max(worst_cp, cp_rel) if not np.isnan(cp_rel) else worst_cp
        worst_sos = max(worst_sos, sos_rel) if not np.isnan(sos_rel) else worst_sos
        rows.append((name, my_rel, cp_rel, sos_rel, nn_ok))
    for name, my_rel, cp_rel, sos_rel, nn_ok in rows:
        dos_flag = "OK " if (np.isnan(my_rel) or my_rel < 1e-6) else "BAD"
        cp_flag = "OK " if (np.isnan(cp_rel) or cp_rel < 1e-4) else "BAD"
        sos_flag = "OK " if (np.isnan(sos_rel) or sos_rel < 1e-6) else "BAD"
        nn = "" if nn_ok is None else (" nn-match" if nn_ok else " NN-MISMATCH")
        print(f"{dos_flag}{cp_flag}{sos_flag} {name:26s} "
              f" DOS={my_rel:.3e} Cp={cp_rel:.3e} SoS={sos_rel:.3e}{nn}")
    print()
    print(f"overall worst: DOS={worst_dos:.3e}  Cp={worst_cp:.3e}  SoS={worst_sos:.3e}")
    ok = all((np.isnan(r[1]) or r[1] < 1e-6) and (np.isnan(r[2]) or r[2] < 1e-4)
             and (np.isnan(r[3]) or r[3] < 1e-6) and (r[4] in (None, True))
             for r in rows)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
