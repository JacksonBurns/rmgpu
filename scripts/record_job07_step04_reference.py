#!/usr/bin/env python3
"""
job-07/step-04 reference capture (RMG-Py, rmg_env).

Builds a toy unimolecular (Lindemann-style) pressure-dependent network and
records EVERYTHING the rmgpu Network (step-04) needs to reproduce it, so the
rmgpu unit test compares rmgpu's OWN computation (grains, eq_ratios, k(E) ILT,
ME matrix, CSE-allen k(T,P)) against RMG-Py's recorded values - non-circular:
the reference is RMG-Py's own run, dumped here, never rmgpu.

Topology (mimics RMG-Py's core-loop no-QM path, rmgpy/rmg/pdep.py):
  isomer A (real conformer -> real DoS)  <==  A --HPL Arrhenius, TS E0 (no QM)-->  product channel B+C (E0 only)
  bath_gas = N2; method = 'chemically-significant eigenvalues' (allen).

Dumped: e_list, j_list, dens_states (isomer + product), Mcoll, Kij/Gnj/Fim
(RMG-Py's rescaled k(E)), eq_ratios, K_ref (RMG-Py's CSE k(T,P)), Tlist/Plist,
grain params, HPL Arrhenius (A,n,Ea SI), TS E0, E0 per channel.

The rmgpu test re-derives: e_list, eq_ratios, k(E) (ILT), the ME matrix, and the
CSE k(T,P) - then compares each to the recorded values.

Run: /home/jackson/miniforge3/envs/rmg_env/bin/python scripts/record_job07_step04_reference.py
"""
import os
import sys
import json
import logging
import numpy as np

logging.disable(logging.CRITICAL)

from rmgpy.pdep.collision import SingleExponentialDown
from rmgpy.pdep.configuration import Configuration
from rmgpy.pdep.network import Network
from rmgpy.reaction import Reaction
from rmgpy.species import Species, TransitionState
from rmgpy.statmech.conformer import Conformer
from rmgpy.statmech.rotation import NonlinearRotor
from rmgpy.statmech.torsion import HinderedRotor
from rmgpy.statmech.translation import IdealGasTranslation
from rmgpy.statmech.vibration import HarmonicOscillator
from rmgpy.transport import TransportData
from rmgpy.kinetics import Arrhenius
from rmgpy.quantity import Quantity
from rmgpy.pdep.reaction import apply_inverse_laplace_transform_method


def _mat2d(a):
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return [[float(v) for v in row] for row in a]


def _tolist(a):
    """Recursively convert a numpy array of any ndim to a nested list of floats."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        return float(a)
    if a.ndim == 1:
        return [float(v) for v in a]
    return [_tolist(row) for row in a]


def build_isomer():
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


def build_product_species():
    B = Species(label="B", conformer=Conformer(E0=(-20.0, "kJ/mol")))
    C = Species(label="C", conformer=Conformer(E0=(-80.0, "kJ/mol")))
    return B, C


def build_bath():
    return Species(label="N2", molecular_weight=(28.04, "g/mol"),
                   transport_data=TransportData(sigma=(3.41, "angstrom"), epsilon=(124, "K")),
                   energy_transfer_model=None)


A_ARR, N_EXP, EA = 1.0e12, 0.5, 30000.0  # s^-1 (unimolecular)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gates", "baselines", "job07", "toy_lindemann_ref.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    A = build_isomer()
    B, C = build_product_species()
    N2 = build_bath()

    kinetics = Arrhenius(A=Quantity(A_ARR, "s^-1"), n=N_EXP, Ea=Quantity(EA, "J/mol"))

    reactant_e0 = A.conformer.E0.value_si
    product_e0 = B.conformer.E0.value_si + C.conformer.E0.value_si
    energy_correction = -min(reactant_e0, product_e0)
    E0_ts = reactant_e0 + EA + energy_correction
    TS = TransitionState(label="TS", conformer=Conformer(E0=(E0_ts * 0.001, "kJ/mol")))
    reaction = Reaction(label="A->BC", reactants=[A], products=[B, C],
                        transition_state=TS, kinetics=kinetics)

    network = Network(
        label="toy-lindemann",
        isomers=[Configuration(A)],
        reactants=[],
        products=[Configuration(B, C)],
        path_reactions=[reaction],
        bath_gas={N2: 1.0},
    )

    Tmin, Tmax, Pmin, Pmax = 300.0, 800.0, 1e3, 1e8
    max_grain_size, min_grain_count = 3000.0, 100
    network.initialize(Tmin, Tmax, Pmin, Pmax, maximum_grain_size=max_grain_size,
                       minimum_grain_count=min_grain_count, rmgmode=True)

    Tlist = [400.0, 600.0]
    Plist = [1e3, 1e4, 1e5, 1e6]
    n_isom = len(network.isomers)
    n_reac = len(network.reactants)
    n_prod = len(network.products)
    fwd = [n_isom + n_reac, 0]

    # Per-(T,P) snapshot. RMG-Py re-selects e_list at each T and recomputes the
    # collision matrix at each (T,P). The collision matrix factors as
    #   Mcoll(T,P) = coll_freq(T,P) * P_coll(T)
    # where P_coll = generate_collision_matrix(...) is P-independent (depends
    # only on T via alpha = alpha0*(T/T0)^n) and coll_freq ~ P. So we store
    # P_coll once per T and coll_freq per (T,P) - lossless, and exactly the
    # interface step-05's collision model will implement.
    snapshots = {}
    Kref = np.zeros((len(Tlist), len(Plist), n_isom+n_reac+n_prod, n_isom+n_reac+n_prod), float)
    for t, T in enumerate(Tlist):
        # The P-independent state at T (set_conditions recomputes it at each P
        # but the values are P-independent except the collision frequency).
        network.set_conditions(T, Plist[0])
        e_init = network.select_energy_grains(T, max_grain_size, min_grain_count)
        block = {
            "e_list": [float(x) for x in network.e_list],
            "j_list": [int(x) for x in network.j_list] if network.j_list is not None else [0],
            "e_list_initial": [float(x) for x in e_init],
            "n_grains_initial": int(len(e_init)),
            "dE_initial": float(e_init[1] - e_init[0]) if len(e_init) > 1 else 0.0,
            "dens_isomer": _mat2d(network.dens_states[0]),
            "dens_product": (_mat2d(network.dens_states[n_isom + n_reac])
                             if network.dens_states[n_isom + n_reac] is not None else None),
            "G_isomer": float(network.isomers[0].get_free_energy(T)),
            "Kij": _tolist(network.Kij),
            "Gnj": _tolist(network.Gnj),
            "Fim": _tolist(network.Fim),
            "eq_ratios": [float(x) for x in network.eq_ratios],
            "n_grains": len(network.e_list),
        }
        # RMG's raw (pre-rescale) ILT k(E) on the stored grid, using the rescaled
        # isomer DoS (as RMG's calculate_microcanonical_rate_coefficient does).
        # Non-circular reference for the rmgpu ILT k(E) port.
        j_arr = np.array([0] if network.j_list is None else network.j_list)
        ilt_ref = apply_inverse_laplace_transform_method(TS, kinetics, network.e_list, j_arr,
                                                         network.dens_states[0], T)
        block["ilt_kE_ref"] = _mat2d(ilt_ref)
        coll_freqs = []
        for p, P in enumerate(Plist):
            network.set_conditions(T, P)
            cf = float(network.coll_freq[0])
            coll_freqs.append(cf)
            Mcoll = np.asarray(network.Mcoll[0], dtype=float)
            block["P_coll"] = _tolist(Mcoll / cf)  # P-independent collision matrix
            network.apply_chemically_significant_eigenvalues_method(method="allen")
            Kref[t, p] = network.K
        block["coll_freqs"] = coll_freqs
        snapshots[str(int(round(T)))] = block

    state = {
        "grain_params": {"Tmin": Tmin, "Tmax": Tmax, "Pmin": Pmin, "Pmax": Pmax,
                          "max_grain_size": max_grain_size, "min_grain_count": min_grain_count},
        "network": {
            "n_isom": n_isom, "n_reac": n_reac, "n_prod": n_prod,
            "E0_isomer": float(reactant_e0),
            "E0_product": float(product_e0),
            "E0_ts": float(E0_ts),
            "energy_correction": float(energy_correction),
        },
        "hpl_arrhenius": {"A": float(A_ARR), "n": float(N_EXP), "Ea": float(EA), "T0": 1.0, "units": "s^-1"},
        "snapshots": snapshots,
        "kTp": {"Tlist": Tlist, "Plist": Plist, "K_ref": _tolist(Kref), "fwd_idx": fwd},
        "method": "chemically-significant eigenvalues",
    }
    with open(out_path, "w") as f:
        json.dump(state, f)

    print("grain_params: Tmax=%.0f max_grain_size=%.0f min_grain_count=%d" % (Tmax, max_grain_size, min_grain_count))
    for t, T in enumerate(Tlist):
        block = snapshots[str(int(round(T)))]
        print("T=%g n_grains=%d dE=%.1f coll_freqs=%s" %
              (T, block["n_grains"], block["e_list"][1]-block["e_list"][0],
               ["%.3e" % c for c in block["coll_freqs"]]))
    print("E0_isomer=%.1f E0_product=%.1f E0_ts=%.1f J/mol" % (reactant_e0, product_e0, E0_ts))
    print("K_ref fwd (isomer->product):")
    for t, T in enumerate(Tlist):
        for p, P in enumerate(Plist):
            print("   T=%g P=%.1e  k=%.5e s^-1" % (T, P, Kref[t, p, fwd[0], fwd[1]]))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
