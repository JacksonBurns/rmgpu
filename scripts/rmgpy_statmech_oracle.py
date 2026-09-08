#!/usr/bin/env python3
"""
job-07/step-07 statmech DoS oracle (RMG-Py, rmg_env).

Builds a set of representative RMG-Py Conformer objects DIRECTLY (no mechanism,
no pdep run - so it is fast) and dumps, for each, the mode parameters AND RMG-
Py's own density of states / sum of states / heat capacity on fixed grids.
This is the NON-CIRCULAR test oracle for porting the production DoS primitives
into rmgpu/statmech/modes.py: the rmgpu port must reproduce these values.

Conformers cover every DoS primitive the production network path uses
(rmgmode=True):
  - HarmonicOscillator  -> Beyer-Swinehart quantum DoS
  - NonlinearRotor      -> classical rigid-rotor DoS
  - LinearRotor         -> classical linear-rotor DoS
  - IdealGasTranslation -> classical translation DoS
  - HinderedRotor       -> classical (elliptic) hindered-rotor DoS
  - FreeRotor           -> free-rotor DoS
  - spin multiplicity (radical)

Run: /home/jackson/miniforge3/envs/rmg_env/bin/python scripts/rmgpy_statmech_oracle.py [out]
"""
import os
import sys
import json
import logging

import numpy as np

logging.disable(logging.CRITICAL)

from rmgpy.statmech.conformer import Conformer
from rmgpy.statmech.vibration import HarmonicOscillator
from rmgpy.statmech.rotation import LinearRotor, NonlinearRotor
from rmgpy.statmech.torsion import HinderedRotor, FreeRotor
from rmgpy.statmech.translation import IdealGasTranslation


def _arr(x):
    if x is None:
        return None
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return float(x)
    if x.ndim == 1:
        return [float(v) for v in x]
    return [_arr(r) for r in x]


def dump_conformer(out, label, e_list, T_list, conf, note=""):
    d = {"label": label, "note": note}
    e0 = conf.E0
    d["E0"] = float(e0.value_si) if e0 is not None else None
    d["spin_multiplicity"] = int(conf.spin_multiplicity)
    d["optical_isomers"] = int(conf.optical_isomers)
    # mode parameters (cm^-1 / amu*angstrom^2 / J/mol) so the assembly can be
    # validated later, and so the rmgpu port reconstructs the same conformer
    modes = []
    for m in conf.modes:
        t = type(m).__name__
        md = {"type": t}
        if t == "HarmonicOscillator":
            # RMG-Py's Frequency.value_si already returns the cm^-1 NUMBER
            # (Frequency(1555,'cm^-1').value_si == 1555.0); do NOT convert.
            md["frequencies_cm1"] = _arr(np.asarray(m.frequencies.value_si, dtype=float))
        elif t == "LinearRotor":
            md["inertia"] = float(m.inertia.value_si)
            md["symmetry"] = int(m.symmetry)
        elif t == "NonlinearRotor":
            md["inertia"] = _arr(np.asarray(m.inertia.value_si, dtype=float))
            md["symmetry"] = int(m.symmetry)
        elif t == "HinderedRotor":
            md["inertia"] = float(m.inertia.value_si)
            md["symmetry"] = int(m.symmetry)
            md["barrier"] = float(m.barrier.value_si) if getattr(m, "barrier", None) is not None else None
        elif t == "FreeRotor":
            md["inertia"] = float(m.inertia.value_si)
            md["symmetry"] = int(m.symmetry)
        elif t == "IdealGasTranslation":
            md["mass"] = float(m.mass.value_si)
        modes.append(md)
    d["modes"] = modes
    d["Cp_T"] = {"T": [float(t) for t in T_list],
                 "values": [float(conf.get_heat_capacity(t)) for t in T_list]}
    try:
        d["DoS"] = {"e": [float(v) for v in e_list],
                    "values": _arr(np.asarray(conf.get_density_of_states(e_list), dtype=float))}
        d["SoS"] = {"e": [float(v) for v in e_list],
                    "values": _arr(np.asarray(conf.get_sum_of_states(e_list), dtype=float))}
    except Exception as e:
        d["error"] = str(e)
    out.append(d)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "gates", "baselines", "propane_branching", "statmech_dos_oracle.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Grids. e_list in J/mol above the ground state (RMG's DoS convention).
    e_list = np.arange(0.0, 40001.0, 1000.0, dtype=float)   # 41 points
    T_list = np.array([300.0, 500.0, 700.0, 1000.0, 1500.0], dtype=float)

    out = []

    # 1. Pure harmonic oscillator set (tests the Beyer-Swinehart quantum DoS).
    conf = Conformer(E0=(0.0, "J/mol"), modes=[
        HarmonicOscillator(frequencies=([250, 500, 800, 1200, 3000, 3200], "cm^-1")),
    ], spin_multiplicity=1, optical_isomers=1)
    dump_conformer(out, "harmonic_only", e_list, T_list, conf,
                   "6 HO; tests Beyer-Swinehart quantum DoS")

    # 2. Translation + nonlinear rotation + HO (a generic polyatomic, closed
    #    shell). This is the core of a real molecule's DoS.
    conf = Conformer(E0=(0.0, "J/mol"), modes=[
        IdealGasTranslation(mass=(44.01, "g/mol")),
        NonlinearRotor(inertia=([50.0, 120.0, 150.0], "amu*angstrom^2"), symmetry=1),
        HarmonicOscillator(frequencies=([400, 700, 1000, 1400, 2800, 3000, 3100], "cm^-1")),
    ], spin_multiplicity=1, optical_isomers=1)
    dump_conformer(out, "polyatomic_closes", e_list, T_list, conf,
                   "translation+nonlinear-rotation+7 HO")

    # 3. Same + a hindered rotor (methyl-style rotor, low barrier) - tests the
    #    classical elliptic hindered-rotor DoS.
    conf = Conformer(E0=(0.0, "J/mol"), modes=[
        IdealGasTranslation(mass=(46.07, "g/mol")),
        NonlinearRotor(inertia=([60.0, 130.0, 160.0], "amu*angstrom^2"), symmetry=1),
        HarmonicOscillator(frequencies=([380, 650, 950, 1300, 2750, 2980], "cm^-1")),
        HinderedRotor(inertia=(2.8, "amu*angstrom^2"), symmetry=3, barrier=(300.0, "cm^-1")),
    ], spin_multiplicity=1, optical_isomers=1)
    dump_conformer(out, "with_hindered_rotor", e_list, T_list, conf,
                   "adds a 3-fold hindered rotor (300 cm^-1 barrier)")

    # 4. Same + a free rotor (high/zero barrier) - tests free-rotor DoS.
    conf = Conformer(E0=(0.0, "J/mol"), modes=[
        IdealGasTranslation(mass=(46.07, "g/mol")),
        NonlinearRotor(inertia=([60.0, 130.0, 160.0], "amu*angstrom^2"), symmetry=1),
        HarmonicOscillator(frequencies=([380, 650, 950, 1300, 2750, 2980], "cm^-1")),
        FreeRotor(inertia=(2.8, "amu*angstrom^2"), symmetry=3),
    ], spin_multiplicity=1, optical_isomers=1)
    dump_conformer(out, "with_free_rotor", e_list, T_list, conf,
                   "adds a free rotor (symmetry 3)")

    # 5. A radical (spin multiplicity 2) - tests the spin factor in the DoS.
    conf = Conformer(E0=(0.0, "J/mol"), modes=[
        IdealGasTranslation(mass=(15.03, "g/mol")),
        NonlinearRotor(inertia=([1.5, 2.0, 3.5], "amu*angstrom^2"), symmetry=1),
        HarmonicOscillator(frequencies=([3100, 3200, 3400, 1450], "cm^-1")),
    ], spin_multiplicity=2, optical_isomers=1)
    dump_conformer(out, "radical_spin2", e_list, T_list, conf,
                   "CH3-like radical, spin multiplicity 2")

    # 6. A linear species (linear rotor) - tests the linear-rotor DoS.
    conf = Conformer(E0=(0.0, "J/mol"), modes=[
        IdealGasTranslation(mass=(28.01, "g/mol")),
        LinearRotor(inertia=(0.86, "amu*angstrom^2"), symmetry=2),
        HarmonicOscillator(frequencies=([2331,], "cm^-1")),
    ], spin_multiplicity=1, optical_isomers=1)
    dump_conformer(out, "linear_N2like", e_list, T_list, conf,
                   "linear species with a linear rotor (N2-like)")

    with open(out_path, "w") as f:
        json.dump({"e_list": [float(v) for v in e_list],
                   "T_list": [float(t) for t in T_list],
                   "n_conformers": len(out),
                   "conformers": out}, f, indent=2)
    print("statmech DoS oracle written:", out_path)
    print("  %d conformers, e_list %d-%d J/mol step %g, T_list %s" %
          (len(out), e_list[0], e_list[-1], e_list[1]-e_list[0],
           [int(t) for t in T_list]))
    for c in out:
        dos = c.get("DoS")
        if dos is not None:
            vals = dos["values"]
            nz = sum(1 for v in vals if v > 0)
            print("  %-20s DoS nonzero %d/%d  max=%.4g  spin=%d" %
                  (c["label"], nz, len(vals), max(vals) if vals else 0,
                   c["spin_multiplicity"]))
        else:
            print("  %-20s ERROR: %s" % (c["label"], c.get("error", "?")))


if __name__ == "__main__":
    main()
