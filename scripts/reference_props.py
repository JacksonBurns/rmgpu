#!/usr/bin/env python
"""RMG-Py reference generator for the job-04 thesis test (gate_04).

Runs in the RMG-Py conda env (rmg_env) - NOT the rmgpu env - because it
exercises RMG-Py's OWN reference paths (thermo libraries + group additivity;
kinetics stored-Arrhenius + family rate rules). It dumps, to
gates/baselines/thesis_test/, the reference Hf298/S298/Cp(T) per species and
k(T) per reaction that the rmgpu gate (gates/gate_04.py) compares its ML
values against.

Commit the two JSON files. They are the ground truth for the thesis test:
the rmgpu gate never re-derives the reference, it reads these.

Species set (gate check 1+2, thermo):
  union of species in examples {superminimal, c3h4, minimal, minimal_ml,
  ethane-oxidation} + all species in primaryThermoLibrary.

Reaction set (gate check 1+2, kinetics):
  the c3h4 mechanism-relevant set (its GRI-Mech3.0-N seed mechanism) + the
  superminimal mechanism-relevant set (its 5 families over the seed species)
  + the depository subset (BurkeH2O2inArHe reaction library).

UNIT CONVENTION (the comparison contract):
  thermo  Hf298 J/mol, S298 J/(mol*K), Cp J/(mol*K) at 300/600/1000 K.
  kinetics k(T) in SI (1/s unimolecular, m^3/(mol*s) bimolecular) - exactly
    what RMG-Py's own get_rate_coefficient returns. The rmgpu ML side emits
    A in CGS cm^3/(mol*s) (per-site, PLAN 3b); the gate converts k_cgs ->
    k_si with 10**(-6*(molecularity-1)) before comparing log10(k_rmgpu/k_rmg).

Usage:
  /home/jackson/miniforge3/envs/rmg_env/bin/python scripts/reference_props.py
"""
from __future__ import annotations

import itertools
import json
import logging
import math
import time
from pathlib import Path

logging.disable(logging.CRITICAL)  # keep the JSON on stdout clean

DB = "/home/jackson/rmgpu/RMG-database/input"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "gates" / "baselines" / "thesis_test"

# The 7 example species (label, SMILES, multiplicity). Union of the 5
# example input.py files: superminimal(H2,O2) c3h4(CH2,C2H2,N2)
# minimal(ethane) minimal_ml(ethane) ethane-oxidation(ethane,O2,Ar).
EXAMPLE_SPECIES = [
    ("H2", "[H][H]", 1),
    ("O2", "[O][O]", 3),
    ("CH2", "[CH2]", 3),
    ("C2H2", "C#C", 1),
    ("N2", "N#N", 1),
    ("ethane", "CC", 1),
    ("Ar", "[Ar]", 1),
]

# The c3h4/superminimal mechanism-relevant reaction buckets.
GRI_LIB = "GRI-Mech3.0-N"      # c3h4 seed mechanism
BURKE_LIB = "BurkeH2O2inArHe"  # a depository-style reaction library
SUPERMINIMAL_FAMILIES = [
    "H_Abstraction", "Disproportionation", "R_Recombination",
    "Birad_recombination", "Birad_R_Recombination",
]
# seed species a superminimal run starts from + the first-generation radicals
# it produces, so the family set reflects a real first-iteration edge.
SUPERMINIMAL_SEED = [
    ("H2", "[H][H]", 1), ("O2", "[O][O]", 3), ("H", "[H]", 1),
    ("O", "[O]", 1), ("OH", "[OH]", 1),
]

T_LIST = (300.0, 600.0, 1000.0)
PER_SPECIES_TIMEOUT_S = 90  # hard cap: one pathological species must not hang the run


class _SpeciesTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _SpeciesTimeout()


def bounded(fn, timeout_s: float = PER_SPECIES_TIMEOUT_S):
    """Run fn() with a hard wall-clock cap (signal.alarm; main thread only)."""
    import signal
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def classify_species(smiles: str) -> str:
    """Stratify: radical (1 unpaired), intermediate (>=2), stable (0).

    A bracket atom ([X] / [X]...) marks an open shell in these SMILES; count
    them. This drives the report's per-class stratification (PLAN 13 risk 2:
    intermediates are the coverage gap that matters most).
    """
    radicals = 0
    i = 0
    while i < len(smiles):
        c = smiles[i]
        if c == "[":
            j = smiles.find("]", i)
            if j > i:
                frag = smiles[i + 1:j]
                # an open-shell atom is bracketed with no explicit bond to
                # fill the valence; heuristic: bracketed C/O/N/S = radical site
                if any(ch.isalpha() for ch in frag):
                    radicals += 1
                i = j
        i += 1
    if radicals >= 2:
        return "intermediate"
    if radicals == 1:
        return "radical"
    return "stable"


def build_species_set(thermo_db):
    """Union of PTL species (by RMG's own to_smiles) + the 7 example species.

    Returns an ordered dict: smiles -> {label, in_ptl, class}.
    """
    from rmgpy.molecule import Molecule  # noqa: F401  (import order check)

    species: dict = {}
    ptl_lib = thermo_db.libraries["primaryThermoLibrary"]
    ptl_smiles = set()
    for label, entry in ptl_lib.entries.items():
        try:
            smi = entry.item.to_smiles()
        except Exception:
            continue
        ptl_smiles.add(smi)
        species.setdefault(smi, {"label": label, "in_ptl": True,
                                 "class": classify_species(smi)})
    for label, smi, _mult in EXAMPLE_SPECIES:
        sp = species.setdefault(smi, {"label": label,
                                      "in_ptl": smi in ptl_smiles,
                                      "class": classify_species(smi)})
        sp["label"] = sp["label"] or label
    return species


def main() -> int:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # A real RMG job constructs RMGDatabase() first (it sets the global
    # input/solvent context). Without it, get_thermo_data's get_input
    # fallback path hangs (reproducible: 60s+ on H2 vs 0.25s with it). So
    # instantiate it up front - this is the realistic reference setup.
    from rmgpy.data.rmg import RMGDatabase
    _rmg_db = RMGDatabase()

    # ------------------------------------------------------------------
    # THERMO reference (libraries + group additivity, no QM)
    # ------------------------------------------------------------------
    from rmgpy.data.thermo import ThermoDatabase
    from rmgpy.species import Species
    from rmgpy.molecule import Molecule

    t0 = time.time()
    tdb = ThermoDatabase()
    tdb.load(DB + "/thermo", libraries=["primaryThermoLibrary"],
             depository=True, surface=False)
    load_s = time.time() - t0
    n_ptl = len(tdb.libraries["primaryThermoLibrary"].entries)

    species = build_species_set(tdb)
    print(f"[thermo] species set: {len(species)} (PTL {n_ptl} + example union), "
          f"load {load_s:.1f}s")

    def compute(smi: str, meta: dict) -> dict:
        mol = Molecule().from_smiles(smi)
        spc = Species(label=meta["label"], molecule=[mol])
        thermo = tdb.get_thermo_data(spc)
        if thermo is None:
            raise RuntimeError("no thermo (neither library nor GA covered)")
        rec = {
            "Hf298": float(thermo.get_enthalpy(298.15)),
            "S298": float(thermo.get_entropy(298.15)),
            "Cp300": float(thermo.get_heat_capacity(300.0)),
            "Cp600": float(thermo.get_heat_capacity(600.0)),
            "Cp1000": float(thermo.get_heat_capacity(1000.0)),
        }
        src = str(getattr(thermo, "comment", ""))
        rec["source"] = "library" if "library" in src.lower() else "GA"
        return rec

    thermo_out = {}
    n_err = 0
    t1 = time.time()
    for n, (smi, meta) in enumerate(species.items(), start=1):
        t_sp = time.time()
        rec = {"label": meta["label"], "in_ptl": meta["in_ptl"],
               "class": meta["class"]}
        try:
            rec.update(bounded(lambda: compute(smi, meta)))
        except _SpeciesTimeout:
            rec["error"] = f"TIMEOUT >{PER_SPECIES_TIMEOUT_S}s"
            n_err += 1
        except Exception as e:  # one bad species must not kill the run
            rec["error"] = f"{type(e).__name__}: {e}"
            n_err += 1
        dt = time.time() - t_sp
        if n % 5 == 1 or n == len(species) or dt > 5:
            status = "err" if "error" in rec else rec.get("source", "?")
            print(f"[thermo] {n}/{len(species)} {smi} {status} "
                  f"({dt:.1f}s, total {time.time()-t1:.1f}s)", flush=True)
        thermo_out[smi] = rec
    per_sp = (time.time() - t1) / max(len(thermo_out), 1)
    print(f"[thermo] {len(thermo_out)} species in {time.time()-t1:.1f}s "
          f"({per_sp:.2f}s/species, {n_err} errors)")

    thermo_json = {
        "description": "RMG-Py reference thermo (primaryThermoLibrary + group "
                       "additivity, no QM) for the job-04 thesis test. "
                       "Hf298 J/mol, S298 J/(mol*K), Cp J/(mol/K) at 300/600/1000 K. "
                       "Generated by scripts/reference_props.py in rmg_env.",
        "database": DB,
        "libraries": ["primaryThermoLibrary"],
        "T_points": list(T_LIST),
        "n_species": len(thermo_out),
        "n_errors": n_err,
        "species": thermo_out,
    }
    (OUT_DIR / "thermo.json").write_text(json.dumps(thermo_json, indent=2))
    print(f"[thermo] wrote {OUT_DIR/'thermo.json'}")

    # ------------------------------------------------------------------
    # KINETICS reference (stored Arrhenius + family rate rules), SI k(T)
    # ------------------------------------------------------------------
    from rmgpy.data.kinetics.database import KineticsDatabase
    from rmgpy.kinetics import Arrhenius

    def reaction_smiles(reactant_mols, product_mols) -> str:
        r = ".".join(sp.to_smiles() for sp in reactant_mols)
        p = ".".join(sp.to_smiles() for sp in product_mols)
        return f"{r}>>{p}"

    kdb = KineticsDatabase()
    kdb.load_recommended_families(DB + "/kinetics/families/recommended.py")
    t0 = time.time()
    kdb.load_families(DB + "/kinetics/families", families="none",
                      depositories=None)
    kdb.load_libraries(DB + "/kinetics/libraries",
                       libraries=[GRI_LIB, BURKE_LIB])
    print(f"[kinetics] loaded {GRI_LIB} + {BURKE_LIB} in {time.time()-t0:.1f}s")

    reactions_out: dict = {}
    bucket_counts = {}

    def add_library_rxn(bucket: str, rxn):
        if not isinstance(rxn.kinetics, Arrhenius):
            return
        if getattr(rxn, "duplicate", False):
            return
        try:
            r_mols = [sp.molecule[0] for sp in rxn.reactants]
            p_mols = [sp.molecule[0] for sp in rxn.products]
            smi = reaction_smiles(r_mols, p_mols)
        except Exception:
            return
        rec = {
            "bucket": bucket,
            "reaction_smiles": smi,
            "reactants": [sp.to_smiles() for sp in r_mols],
            "products": [sp.to_smiles() for sp in p_mols],
            "molecularity": len(rxn.reactants),
            "degeneracy": float(getattr(rxn, "degeneracy", 1) or 1),
            "source": "stored-Arrhenius",
        }
        for T in T_LIST:
            try:
                rec[f"k{int(T)}"] = float(rxn.kinetics.get_rate_coefficient(T))
            except Exception:
                rec[f"k{int(T)}"] = None
        if smi not in reactions_out:
            reactions_out[smi] = rec
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    for lib_name in (GRI_LIB, BURKE_LIB):
        for rxn in kdb.libraries[lib_name].get_library_reactions():
            add_library_rxn(lib_name, rxn)
    print(f"[kinetics] stored-Arrhenius: { {k: v for k, v in bucket_counts.items()} }")

    # superminimal mechanism-relevant set: the 5 families over the seed species
    t0 = time.time()
    kdb2 = KineticsDatabase()
    kdb2.load_recommended_families(DB + "/kinetics/families/recommended.py")
    kdb2.load_families(DB + "/kinetics/families",
                       families=SUPERMINIMAL_FAMILIES, depositories=["training"])

    def make_mol(smi, mult):
        m = Molecule().from_smiles(smi)
        m.multiplicity = mult
        m.reactive = True
        return m

    seed_mols = [make_mol(s, m) for _l, s, m in SUPERMINIMAL_SEED]
    pairs = [list(p) for p in itertools.combinations_with_replacement(seed_mols, 2)]
    pairs.append([make_mol("[H]", 1), make_mol("[H]", 1), make_mol("[O][O]", 3)])
    pairs.append([make_mol("[O]", 1), make_mol("[O]", 1), make_mol("[H][H]", 1)])

    for fam_label in SUPERMINIMAL_FAMILIES:
        fam = kdb2.families[fam_label]
        for p in pairs:
            try:
                rxns = fam.generate_reactions(p, products=None,
                                              prod_resonance=False,
                                              delete_labels=True,
                                              relabel_atoms=True)
            except Exception:
                continue
            for r in rxns:
                if getattr(r, "duplicate", False):
                    continue
                try:
                    k_obj, _src, _entry, _is_fwd = fam.get_kinetics(
                        r, template_labels=list(r.template or []),
                        degeneracy=1, estimator="rate rules",
                        return_all_kinetics=False)
                    if k_obj is None:
                        continue
                except Exception:
                    continue
                try:
                    smi = reaction_smiles(r.reactants, r.products)
                except Exception:
                    continue
                rec = {
                    "bucket": f"superminimal:{fam_label}",
                    "reaction_smiles": smi,
                    "reactants": [sp.to_smiles() for sp in r.reactants],
                    "products": [sp.to_smiles() for sp in r.products],
                    "molecularity": len(r.reactants),
                    "degeneracy": 1,
                    "source": "rate-rules",
                }
                for T in T_LIST:
                    try:
                        rec[f"k{int(T)}"] = float(k_obj.get_rate_coefficient(T, 0))
                    except Exception:
                        rec[f"k{int(T)}"] = None
                if smi not in reactions_out:
                    reactions_out[smi] = rec
                    bucket_counts[f"superminimal:{fam_label}"] = \
                        bucket_counts.get(f"superminimal:{fam_label}", 0) + 1
    print(f"[kinetics] superminimal family set in {time.time()-t0:.1f}s")

    reactions_json = {
        "description": "RMG-Py reference HPL k(T) for the job-04 thesis test. "
                       "Buckets: GRI-Mech3.0-N (c3h4 seed), BurkeH2O2inArHe "
                       "(depository), superminimal:<family> (rate rules). "
                       "k(T) in SI (get_rate_coefficient): 1/s unimolecular, "
                       "m^3/(mol*s) bimolecular. The rmgpu ML side emits A in "
                       "CGS cm^3/(mol*s); the gate converts k_cgs->k_si with "
                       "10**(-6*(molecularity-1)) before log10(k_rmgpu/k_rmg). "
                       "Generated by scripts/reference_props.py in rmg_env.",
        "database": DB,
        "T_points": list(T_LIST),
        "n_reactions": len(reactions_out),
        "bucket_counts": bucket_counts,
        "reactions": reactions_out,
    }
    (OUT_DIR / "reactions.json").write_text(json.dumps(reactions_json, indent=2))
    print(f"[kinetics] wrote {OUT_DIR/'reactions.json'} ({len(reactions_out)} reactions)")

    print(f"[done] {time.time()-t_start:.1f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
