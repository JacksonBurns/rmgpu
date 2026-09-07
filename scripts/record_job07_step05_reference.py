#!/usr/bin/env python3
"""
job-07/step-05 reference capture (RMG-Py, rmg_env).

Builds the SAME toy unimolecular Lindemann network as step-04 (scripts/
record_job07_step04_reference.py) and records the COLLISION-SIDE values the
step-05 rmgpu/pdep/collision.py must reproduce, non-circularly. The step-04
baseline (gates/baselines/job07/toy_lindemann_ref.json) already holds the
per-T collision matrix P_coll, the coll_freq per (T,P), and RMG-Py's CSE (Allen)
k(T,P) K_ref -- those are reused by the full-pipeline test. This script adds the
pieces that baseline does NOT hold:

  1. Collision frequency with a MULTI-species bath (N2 0.5 / Ar 0.5) at several
     (T,P): a real formula check of RMG's LJ bath-averaging (sigma linear,
     epsilon geometric, mass linear), beyond the single-bath identity in the
     step-04 baseline. The single-bath (N2) case is also re-recorded here so the
     test is self-contained.
  2. Collision efficiency: the MSC Chang-Bozzelli-Dean factor
     (SingleExponentialDown.calculate_collision_efficiency) for the toy isomer
     at several (T, e_reac) -- a non-circular ref for the MSC method (job-08).

The georgievskii CSE variant is a job-08 deliverable (NOT ported in step-05);
its RMG-Py implementation (cse.pyx get_rate_coefficients_CSE_Advanced) was
observed to hit an internal index error on this toy topology (a bimolecular
product channel + exclude_association), so it is deliberately NOT recorded here.

Run: /home/jackson/miniforge3/envs/rmg_env/bin/python scripts/record_job07_step05_reference.py
"""
import os
import sys
import json
import logging
import numpy as np

logging.disable(logging.CRITICAL)

from rmgpy.pdep.collision import SingleExponentialDown
from rmgpy.pdep.configuration import Configuration
from rmgpy.species import Species
from rmgpy.statmech.conformer import Conformer
from rmgpy.statmech.rotation import NonlinearRotor
from rmgpy.statmech.torsion import HinderedRotor
from rmgpy.statmech.translation import IdealGasTranslation
from rmgpy.statmech.vibration import HarmonicOscillator
from rmgpy.transport import TransportData


def build_isomer():
    # Identical to scripts/record_job07_step04_reference.py build_isomer().
    return Species(
        label="A",
        conformer=Conformer(
            E0=(-100.0, "kJ/mol"),
            modes=[
                IdealGasTranslation(mass=(74.07, "g/mol")),
                NonlinearRotor(inertia=([41.5091, 215.751, 233.258], "amu*angstrom^2"), symmetry=1),
                HarmonicOscillator(frequencies=(
                    [240.915, 341.933, 500.066, 728.41, 809.987, 833.93, 926.308, 948.571,
                     1009.3, 1031.46, 1076, 1118.4, 1184.66, 1251.36, 1314.36, 1321.42,
                     1381.17, 1396.5, 1400.54, 1448.08, 1480.18, 1485.34, 1492.24, 1494.99,
                     1586.16, 2949.01, 2963.03, 2986.19, 2988.1, 2995.27, 3026.03, 3049.05,
                     3053.47, 3054.83, 3778.88], "cm^-1")),
                HinderedRotor(inertia=(2.81525, "amu*angstrom^2"), symmetry=3, barrier=(2.96807, "kcal/mol")),
            ],
            spin_multiplicity=1, optical_isomers=1,
        ),
        molecular_weight=(74.07, "g/mol"),
        transport_data=TransportData(sigma=(5.94, "angstrom"), epsilon=(559, "K")),
        energy_transfer_model=SingleExponentialDown(alpha0=(447.5 * 0.011962, "kJ/mol"), T0=(300, "K"), n=0.85),
    )


def build_bath(label, mw, sigma, epsilon):
    return Species(label=label, molecular_weight=(mw, "g/mol"),
                   transport_data=TransportData(sigma=(sigma, "angstrom"), epsilon=(epsilon, "K")))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gates", "baselines", "job07", "collision_ref.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    A = build_isomer()
    N2 = build_bath("N2", 28.04, 3.41, 124.0)
    AR = build_bath("Ar", 39.948, 3.405, 93.3)
    etm = A.energy_transfer_model  # SingleExponentialDown

    # --- 1. Collision frequency: single-bath (N2) + multi-bath (N2/Ar 0.5/0.5) ---
    Tlist = [300.0, 400.0, 500.0, 600.0]
    Plist = [1e3, 1e4, 1e5, 1e6]
    freq_single = []
    freq_multi = []
    for T in Tlist:
        for P in Plist:
            cfg = Configuration(A)
            f1 = float(cfg.calculate_collision_frequency(T, P, {N2: 1.0}))
            f2 = float(cfg.calculate_collision_frequency(T, P, {N2: 0.5, AR: 0.5}))
            freq_single.append({"T": T, "P": P, "value": f1})
            freq_multi.append({"T": T, "P": P, "value": f2})

    # --- 2. Collision efficiency (MSC Chang-Bozzelli-Dean factor) ---
    # RMG's MSC path (msc.pyx) calls SingleExponentialDown
    # .calculate_collision_efficiency(T, network.e_list, network.j_list,
    # network.dens_states[i], E0[i], e_reac[i]) for each isomer. We reproduce
    # that EXACTLY: build the step-04 toy network, set conditions at each T,
    # and record the efficiency on the network's real e_list / dens_states.
    from rmgpy.pdep.network import Network as _Net
    from rmgpy.reaction import Reaction
    from rmgpy.species import TransitionState
    from rmgpy.kinetics import Arrhenius
    from rmgpy.quantity import Quantity
    B = Species(label="B", conformer=Conformer(E0=(-20.0, "kJ/mol")))
    C = Species(label="C", conformer=Conformer(E0=(-80.0, "kJ/mol")))
    reactant_e0 = A.conformer.E0.value_si
    product_e0 = B.conformer.E0.value_si + C.conformer.E0.value_si
    energy_correction = -min(reactant_e0, product_e0)
    EA = 30000.0
    E0_ts = reactant_e0 + EA + energy_correction
    TS = TransitionState(label="TS", conformer=Conformer(E0=(E0_ts * 0.001, "kJ/mol")))
    reaction = Reaction(label="A->BC", reactants=[A], products=[B, C],
                        transition_state=TS,
                        kinetics=Arrhenius(A=Quantity(1.0e12, "s^-1"), n=0.5,
                                           Ea=Quantity(EA, "J/mol")))
    network = _Net(label="toy-lindemann", isomers=[Configuration(A)], reactants=[],
                   products=[Configuration(B, C)], path_reactions=[reaction],
                   bath_gas={N2: 1.0})
    network.initialize(300.0, 800.0, 1e3, 1e8, maximum_grain_size=3000.0,
                       minimum_grain_count=100, rmgmode=True)
    E0_isomer = float(A.conformer.E0.value_si)  # J/mol
    eff = []
    for T in [300.0, 400.0, 600.0]:
        network.set_conditions(T, 1e5)
        e_list = np.array(network.e_list, dtype=float)
        dens = np.asarray(network.dens_states[0], dtype=float)  # (n, n_j)
        j_list = np.array([0] if network.j_list is None else network.j_list, dtype=int)
        # e_reac = E0 + offset; a spread of offsets so beta spans its range.
        eff_T = {}
        for offset in [2000.0, 10000.0, 50000.0, 100000.0, 140000.0]:
            e_reac = E0_isomer + offset
            eff_T["e_reac_offset_%g" % offset] = float(
                etm.calculate_collision_efficiency(T, e_list, j_list, dens, E0_isomer, e_reac))
        eff.append({"T": T, "eff": eff_T, "E0_isomer": E0_isomer,
                    "n_grains": len(e_list), "dE": float(e_list[1] - e_list[0]),
                    "e_min": float(e_list[0]), "e_max": float(e_list[-1]),
                    "e_list": [float(x) for x in e_list],
                    "j_list": [int(x) for x in j_list],
                    "dens": [[float(v) for v in row] for row in dens]})

    state = {
        "collision_frequency": {
            "single_bath_N2": freq_single,
            "multi_bath_N2Ar": freq_multi,
            "bath": {
                "N2": {"sigma_angstrom": 3.41, "epsilon_K": 124.0, "mw_g_mol": 28.04},
                "Ar": {"sigma_angstrom": 3.405, "epsilon_K": 93.3, "mw_g_mol": 39.948},
            },
            "species": {"sigma_angstrom": 5.94, "epsilon_K": 559.0, "mw_g_mol": 74.07},
        },
        "collision_efficiency": eff,
        "etm": {"alpha0_J_mol": float(etm.get_alpha(300.0)), "T0": 300.0, "n": 0.85},
    }
    with open(out_path, "w") as f:
        json.dump(state, f)
    print("wrote", out_path)
    print("single-bath N2  f(400,1e3)=%.6e" % freq_single[4]["value"])
    print("multi-bath N2/Ar f(400,1e3)=%.6e" % freq_multi[4]["value"])
    print("collision_efficiency T=300 e_reac+20000:", eff[0]["eff"])


if __name__ == "__main__":
    main()
