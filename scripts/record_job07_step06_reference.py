#!/usr/bin/env python3
"""
job-07/step-06 reference capture (RMG-Py, rmg_env).

Builds a TOY 2-ISOMER pressure-dependent network and records EVERYTHING the
rmgpu pdep driver (step-06) needs to reproduce it, so the driver's unit test
compares rmgpu's OWN computation (network solve -> CSE k(T,P), the Falloff fit,
the pdep/<network>.yaml round-trip) against RMG-Py's recorded values -
NON-CIRCULAR: the reference is RMG-Py's own run, never rmgpu.

Topology (mimics RMG-Py's core-loop isomerization network, rmgpy/rmg/pdep.py):
  isomer A  <==  A --isomerization, HPL Arrhenius + TS E0 (no QM)-->  A2
  A  --dissociation, HPL Arrhenius + TS E0 (no QM)-->  B + C
  bath_gas = N2; method = 'chemically-significant eigenvalues' (allen).
  n_isom=2, n_reac=0, n_prod=1, n_cfg=3.

This is the WIRING test network: it has TWO isomers (so the driver's
multi-isomer state seam + the net-reaction (from,to) pairing + the Falloff
attach are all exercised, not just the single-isomer toy from step-04).

Dumped per T: e_list, j_list, dens_isomers (n_isom, n_grains, n_j),
dens_product, Kij/Gnj/Fim, eq_ratios, P_coll (n_isom, n_g, n_j, n_g, n_j),
coll_freqs (n_P, per isomer) -> so P_coll is P-independent and coll_freq ~ P
(the step-05 collision factorization Mcoll(T,P)=coll_freq(T,P)*P_coll(T)).
Also K_ref: RMG-Py's CSE k(T,P) on the (T,P) grid, and the (from,to) net
reaction pairs + their n_reactants for the fit.

Run: /home/jackson/miniforge3/envs/rmg_env/bin/python scripts/record_job07_step06_reference.py
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


def _mat2d(a):
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return [[float(v) for v in row] for row in a]


def _tolist(a):
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        return float(a)
    if a.ndim == 1:
        return [float(v) for v in a]
    return [_tolist(row) for row in a]


def _conformer(e0_kj, freqs):
    return Conformer(
        E0=(e0_kj, "kJ/mol"),
        modes=[
            IdealGasTranslation(mass=(74.07, "g/mol")),
            NonlinearRotor(inertia=([41.5091, 215.751, 233.258], "amu*angstrom^2"),
                           symmetry=1),
            HarmonicOscillator(frequencies=(freqs, "cm^-1")),
            HinderedRotor(inertia=(2.81525, "amu*angstrom^2"), symmetry=3,
                           barrier=(2.96807, "kcal/mol")),
        ],
        spin_multiplicity=1, optical_isomers=1,
    )


FREQS_A = [240.915, 341.933, 500.066, 728.41, 809.987, 833.93, 926.308, 948.571,
           1009.3, 1031.46, 1076, 1118.4, 1184.66, 1251.36, 1314.36, 1321.42,
           1381.17, 1396.5, 1400.54, 1448.08, 1480.18, 1485.34, 1492.24, 1494.99,
           1586.16, 2949.01, 2963.03, 2986.19, 2988.1, 2995.27, 3026.03, 3049.05,
           3053.47, 3054.83, 3778.88]
FREQS_A2 = [f * 0.998 for f in FREQS_A]  # a slightly softer 2nd isomer


def build_isomer(label, e0_kj, freqs):
    return Species(
        label=label,
        conformer=_conformer(e0_kj, freqs),
        molecular_weight=(74.07, "g/mol"),
        transport_data=TransportData(sigma=(5.94, "angstrom"), epsilon=(559, "K")),
        energy_transfer_model=SingleExponentialDown(
            alpha0=(447.5 * 0.011962, "kJ/mol"), T0=(300, "K"), n=0.85),
    )


def build_products():
    B = Species(label="B", conformer=Conformer(E0=(-20.0, "kJ/mol")))
    C = Species(label="C", conformer=Conformer(E0=(-80.0, "kJ/mol")))
    return B, C


def build_bath():
    return Species(label="N2", molecular_weight=(28.04, "g/mol"),
                   transport_data=TransportData(sigma=(3.41, "angstrom"), epsilon=(124, "K")),
                   energy_transfer_model=None)


# HPL barriers (SI J/mol): raised so the CSE eigenvalue separation stays
# clean across the whole (T,P) grid (low barriers -> too-fast k at the hot
# corner -> CSE under-identifies the chemical eigenvalues and returns zero,
# which is a genuine CSE limitation, not a wiring bug - keep the grid where
# CSE is robust for this WIRING test).
ISO_ARR, ISO_N, ISO_EA = 1.0e11, 0.5, 70000.0
DISS_ARR, DISS_N, DISS_EA = 1.0e12, 0.5, 60000.0


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "gates", "baselines", "job07", "toy_2isomer_ref.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    A = build_isomer("A", -100.0, FREQS_A)
    A2 = build_isomer("A2", -95.0, FREQS_A2)
    B, C = build_products()
    N2 = build_bath()

    energy_correction = -min(A.conformer.E0.value_si, A2.conformer.E0.value_si,
                             B.conformer.E0.value_si + C.conformer.E0.value_si)
    e_a = A.conformer.E0.value_si
    e_a2 = A2.conformer.E0.value_si

    # isomerization A<=>A2
    e0_iso = e_a + ISO_EA + energy_correction
    ts_iso = TransitionState(label="TS_iso",
                             conformer=Conformer(E0=(e0_iso * 0.001, "kJ/mol")))
    rxn_iso = Reaction(label="A<=>A2", reactants=[A], products=[A2],
                       transition_state=ts_iso,
                       kinetics=Arrhenius(A=Quantity(ISO_ARR, "s^-1"), n=ISO_N,
                                          Ea=Quantity(ISO_EA, "J/mol")))
    # dissociation A->B+C
    e0_diss = e_a + DISS_EA + energy_correction
    ts_diss = TransitionState(label="TS_diss",
                              conformer=Conformer(E0=(e0_diss * 0.001, "kJ/mol")))
    rxn_diss = Reaction(label="A->B+C", reactants=[A], products=[B, C],
                        transition_state=ts_diss,
                        kinetics=Arrhenius(A=Quantity(DISS_ARR, "s^-1"), n=DISS_N,
                                           Ea=Quantity(DISS_EA, "J/mol")))

    network = Network(
        label="toy-2isomer",
        isomers=[Configuration(A), Configuration(A2)],
        reactants=[],
        products=[Configuration(B, C)],
        path_reactions=[rxn_iso, rxn_diss],
        bath_gas={N2: 1.0},
    )

    Tmin, Tmax, Pmin, Pmax = 300.0, 800.0, 1e3, 1e6
    max_grain_size, min_grain_count = 3000.0, 100
    network.initialize(Tmin, Tmax, Pmin, Pmax, maximum_grain_size=max_grain_size,
                       minimum_grain_count=min_grain_count, rmgmode=True)

    # Gauss-Chebyshev T/P grid (RMG's Chebyshev interpolation grid) - the
    # driver (step-06) generates the SAME grid from the pressure_dependence
    # block, so the reference is recorded on that exact grid.
    Tcount, Pcount = 8, 6
    Tlist = np.zeros(Tcount)
    for i in range(Tcount):
        x = -np.cos((2 * i + 1) * np.pi / (2 * Tcount))
        Tlist[i] = 2.0 / ((1.0 / Tmax - 1.0 / Tmin) * x + 1.0 / Tmax + 1.0 / Tmin)
    Plist = np.zeros(Pcount)
    for i in range(Pcount):
        x = -np.cos((2 * i + 1) * np.pi / (2 * Pcount))
        Plist[i] = 10 ** (0.5 * ((np.log10(Pmax) - np.log10(Pmin)) * x
                                 + np.log10(Pmax) + np.log10(Pmin)))
    Tlist = np.sort(Tlist)
    Plist = np.sort(Plist)

    n_isom = 2
    n_reac = 0
    n_prod = 1
    n_cfg = n_isom + n_reac + n_prod
    Kref = np.zeros((len(Tlist), len(Plist), n_cfg, n_cfg), float)
    snapshots = {}
    for t, T in enumerate(Tlist):
        network.set_conditions(T, Plist[0])
        dens_isomers = np.asarray(network.dens_states[:n_isom], dtype=float)
        block = {
            "e_list": [float(x) for x in network.e_list],
            "j_list": [int(x) for x in network.j_list] if network.j_list is not None else [0],
            "dens_isomers": _tolist(dens_isomers),
            "dens_product": (_mat2d(network.dens_states[n_isom + n_reac])
                             if network.dens_states[n_isom + n_reac] is not None else None),
            "Kij": _tolist(network.Kij),
            "Gnj": _tolist(network.Gnj),
            "Fim": _tolist(network.Fim),
            "eq_ratios": [float(x) for x in network.eq_ratios],
            "n_grains": len(network.e_list),
        }
        coll_freqs = []
        for p, P in enumerate(Plist):
            network.set_conditions(T, P)
            cf = np.asarray(network.coll_freq, dtype=float)
            coll_freqs.append([float(c) for c in cf])
            # per-isomer P-independent collision matrix
            pcoll = np.zeros((n_isom,) + network.Mcoll[0].shape, float)
            for i in range(n_isom):
                if cf[i] != 0:
                    pcoll[i] = np.asarray(network.Mcoll[i], dtype=float) / cf[i]
            block["P_coll"] = _tolist(pcoll)
            network.apply_chemically_significant_eigenvalues_method(method="allen")
            Kref[t, p] = network.K
        block["coll_freqs"] = coll_freqs
        snapshots[str(round(float(T), 6))] = block

    # The net reactions to fit (RMG fit_interpolation_models loop): for
    # prod in range(n_cfg): for reac in range(n_isom+n_reac): prod!=reac.
    net_reactions = []
    for prod in range(n_cfg):
        for reac in range(n_isom + n_reac):
            if prod == reac:
                continue
            # n_reactants: number of species in the `reac` configuration
            if reac < n_isom:
                n_reactants = 1  # an isomer
            else:
                n_reactants = len(network.reactants[reac - n_isom].species)
            net_reactions.append({"from_cfg": reac, "to_cfg": prod,
                                  "n_reactants": n_reactants})

    # Wiring-test sanity: the forward channels (isomer -> product) must be
    # positive on the WHOLE grid - a CSE zero-free grid keeps the log-space
    # fit clean (CSE returns zero where the chemical eigenvalues aren't
    # separated; the grid is chosen to avoid that).
    for (r, c) in [(1, 0), (2, 0), (2, 1)]:
        if not np.all(Kref[:, :, r, c] > 0):
            raise SystemExit(
                "CSE returned zeros on the grid for channel (%d->%d): "
                "31/48-style sparse solve. Widen the barriers or shrink the "
                "grid so the WIRING test has a clean, zero-free solve." % (c, r))

    state = {
        "grain_params": {"Tmin": float(Tmin), "Tmax": float(Tmax),
                         "Pmin": float(Pmin), "Pmax": float(Pmax),
                         "max_grain_size": float(max_grain_size),
                         "min_grain_count": int(min_grain_count)},
        "network": {"n_isom": n_isom, "n_reac": n_reac, "n_prod": n_prod,
                    "E0_isomers": [e_a, e_a2],
                    "E0_product": float(B.conformer.E0.value_si + C.conformer.E0.value_si),
                    "energy_correction": float(energy_correction)},
        "hpl_arrhenius": {
            "iso": {"A": float(ISO_ARR), "n": float(ISO_N), "Ea": float(ISO_EA), "T0": 1.0},
            "diss": {"A": float(DISS_ARR), "n": float(DISS_N), "Ea": float(DISS_EA), "T0": 1.0},
        },
        "snapshots": snapshots,
        "kTp": {"Tlist": [float(t) for t in Tlist], "Plist": [float(p) for p in Plist],
                "K_ref": _tolist(Kref)},
        "net_reactions": net_reactions,
        "method": "chemically-significant eigenvalues",
    }
    with open(out_path, "w") as f:
        json.dump(state, f)
    print("Tlist:", [round(float(t), 2) for t in Tlist])
    print("Plist:", ["%.3e" % p for p in Plist])
    print("n_cfg=%d net_reactions=%d" % (n_cfg, len(net_reactions)))
    # a couple of K_ref probes (isomer A -> product, A -> A2)
    print("K_ref[0,0] (A->A2)=%.4e  (A->BC)=%.4e  (A2->BC)=%.4e" %
          (Kref[0, 0, 1, 0], Kref[0, 0, 2, 0], Kref[0, 0, 2, 1]))
    print("K_ref[-1,-1] (A->A2)=%.4e  (A->BC)=%.4e  (A2->BC)=%.4e" %
          (Kref[-1, -1, 1, 0], Kref[-1, -1, 2, 0], Kref[-1, -1, 2, 1]))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
