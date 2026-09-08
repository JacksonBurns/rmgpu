"""Job driver for rmgpu (job-06, the "moment of truth").

``run(input_path)`` loads the resolved YAML input, builds the databases
(job 02), the ML estimators (job 04), the reaction families (job 05), the
seed species, and runs the core/edge iteration loop (job 06) to steady state,
then writes the output tree (PLAN 12.3):

    run/
      run.yaml  provenance.yaml  summary.md  rmgpu.log  events.jsonl
      mechanism/core.yaml  mechanism/edge.yaml
      species/<label>.json  reactions/reactions.json
      profiles/<reactor>/time_series.csv + metadata.yaml

The loop is deterministic (sorted family / species ordering); the pdep hook
is the HPL stub (job 07 turns pdep on in both rmgpu and the reference).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import yaml

from rmgpu.schemas.input import Input, resolve_extends
from rmgpu.units import Quantity
from rmgpu.core.model import Species
from rmgpu.core.loop import CoreEdgeLoop, LoopConfig, RunContext
from rmgpu.molecule.molecule import Molecule
from rmgpu.logging import log, setup_logging

# Quantity YAML representer (mirrors cli.py)
class _QDumper(yaml.SafeDumper):
    pass

def _rep_q(dumper, q: Quantity):
    return dumper.represent_mapping(
        'tag:yaml.org,2002:map', {'value': q._value, 'unit': q._units})

_QDumper.add_representer(Quantity, _rep_q)


# ---------------------------------------------------------------------------
# Input loading / block construction
# ---------------------------------------------------------------------------

def load_input(path: str) -> Input:
    path = os.path.abspath(path)
    base_dir = os.path.dirname(path)
    log.info("Loading input %s", path)
    with open(path) as f:
        doc = yaml.safe_load(f)
    doc = resolve_extends(doc, base_dir)
    return Input(**doc)


def _smiles_of(struct) -> str:
    if isinstance(struct, str):
        return struct
    # Support both SMILES and InChI
    smiles = getattr(struct, "smiles", None)
    inchi = getattr(struct, "inchi", None)
    # StructureValue.value returns the first non-None field
    value = getattr(struct, "value", None)
    if smiles:
        return smiles
    if inchi:
        return f"inchi:{inchi}"
    if value:
        # If value looks like an InChI, treat as inchi
        if isinstance(value, str) and value.startswith("InChI="):
            return f"inchi:{value}"
        return value
    return ""


def _build_seed_species(model_input: Input) -> list[Species]:
    log.info("Building seed species from input")
    out = []
    for sp in model_input.species or []:
        val = _smiles_of(sp.structure)
        if not val:
            raise ValueError(f"Species {sp.label} has no SMILES/InChI")
        if isinstance(val, str) and val.startswith("inchi:"):
            inchi = val[6:]
            out.append(Species(label=sp.label, molecule=Molecule(inchi=inchi),
                               reactive=bool(sp.reactive)))
        else:
            out.append(Species(label=sp.label, molecule=Molecule(smiles=val),
                               reactive=bool(sp.reactive)))
    return out


def _build_seed_mechanisms(model_input: Input, databases):
    """Load the requested seed mechanisms from rmgdb.

    A load failure is LAUD, not swallowed: a run that asked for a seed
    mechanism and got none is not a valid run, so the exception propagates
    and the run FAILs (job-06/step-07, Fix 3). Per-mechanism loud summaries
    (resolution, drops) are returned for the run summary.
    """
    from rmgpu.db.seed_loader import load_seed_mechanism
    db_block = getattr(model_input, "database", None)
    seed_names = []
    if db_block is not None:
        seed_names = list(db_block.seed_mechanisms or [])
    all_species = []
    all_reactions = []
    seen_species_keys = set()
    seen_reaction_keys = set()
    summaries = []
    for name in seed_names:
        log.info("Loading seed mechanism %r", name)
        # Name resolution (RMG-Py name -> rmgdb library name) lives inside the
        # loader and is loud (alias table + controlled normalization).
        species, reactions, summary = load_seed_mechanism(name, databases)
        summary["requested"] = name
        summaries.append(summary)
        log.info("Seed %r resolved to %r (%s): %d species / %d reactions loaded",
                 name, summary.get("resolved_library_name"), summary.get("resolution"),
                 summary.get("n_species", 0), summary.get("n_reactions", 0))
        # Deduplicate by canonical SMILES
        for sp in species:
            from rmgpu.core.loop import canonical_key
            key = canonical_key(sp.molecule)
            if key not in seen_species_keys:
                seen_species_keys.add(key)
                all_species.append(sp)
        for rx in reactions:
            # Build a simple key from reactant/product labels
            r_labels = tuple(sorted(s.label for s in rx.reactants))
            p_labels = tuple(sorted(s.label for s in rx.products))
            key = (r_labels, p_labels)
            if key not in seen_reaction_keys:
                seen_reaction_keys.add(key)
                all_reactions.append(rx)
    return all_species, all_reactions, summaries


def _build_databases(model_input: Input):
    from rmgpu.db import Databases
    log.info("Building databases from input")
    db_block = getattr(model_input, "database", None)
    if db_block is None:
        log.info("No database block in input – using empty Databases")
        return Databases.from_config({})
    cfg = {
        "thermo_libraries": list(db_block.thermo_libraries or []),
        "reaction_libraries": list(db_block.reaction_libraries or []),
        "kinetics_families": db_block.kinetics_families,
        "seed_mechanisms": list(db_block.seed_mechanisms or []),
    }
    log.debug("Database config: %s", cfg)
    db = Databases.from_config(cfg)
    log.info("Databases built")
    return db


def _build_ml():
    """Build the ML estimators. If a checkpoint is missing on the box the
    estimators are BLOCKED (None) - every ML hit becomes a recorded coverage
    finding, the run completes (never a silent fallback, never a crash)."""
    from pathlib import Path
    from rmgpu.ml.base import MODELS_DIR
    from rmgpu.ml.thermo_estimator import ThermoML
    from rmgpu.ml.kinetics_estimator import KineticsML

    log.info("Building ML estimators")
    class _ML:
        thermo = None
        kinetics = None

    ml = _ML()
    blocked = []
    thermo_ckpt = Path(MODELS_DIR) / "chemeleon_thermo_662946.ckpt"
    kin_ckpt = Path(MODELS_DIR) / "chemprop_kinetics_662946.ckpt"
    if thermo_ckpt.exists():
        try:
            log.debug("Loading ThermoML from %s", MODELS_DIR)
            ml.thermo = ThermoML(MODELS_DIR)
            log.info("ThermoML loaded")
        except Exception:
            log.exception("ThermoML load failed")
            blocked.append("thermo")
    else:
        log.warning("Thermo checkpoint missing")
        blocked.append("thermo")
    if kin_ckpt.exists():
        try:
            log.debug("Loading KineticsML from %s", MODELS_DIR)
            ml.kinetics = KineticsML(MODELS_DIR)
            log.info("KineticsML loaded")
        except Exception:
            log.exception("KineticsML load failed")
            blocked.append("kinetics")
    else:
        log.warning("Kinetics checkpoint missing")
        blocked.append("kinetics")
    ml.blocked = blocked
    log.info("ML estimators built – blocked: %s", blocked)
    return ml


def _build_families(families_sel):
    from rmgpu.core.family import KineticsFamilies
    log.info("Building families – selection=%s", families_sel)
    kf = KineticsFamilies()
    if families_sel in ("default", "all"):
        kf.load(families_sel)
    elif isinstance(families_sel, str):
        kf.load([families_sel])
    elif isinstance(families_sel, (list, tuple)) and families_sel:
        kf.load(list(families_sel))
    else:
        kf.load("default")
    log.info("Families loaded")
    return kf


def _build_reactor(model_input: Input):
    """Return (temperature_K, pressure_Pa, initial_mole_fractions,
    termination_time_s, termination_conversion) for the first reactor."""
    reactors = getattr(model_input, "reactors", None) or []
    if not reactors:
        raise ValueError("no reactors in the input")
    r = reactors[0]
    T = float(r.temperature.to_si())
    P = float(r.pressure.to_si())
    imf = {k: float(v) for k, v in (r.initial_mole_fractions or {}).items()}
    t_term = None
    conv = None
    term = getattr(r, "termination", None)
    if term is not None:
        t = getattr(term, "time", None)
        if t is not None:
            t_term = float(t.to_si())
        c = getattr(term, "conversion", None)
        if c is not None:
            try:
                conv = {c["species"]: float(c["value"])}
            except Exception:  # noqa: BLE001
                conv = None
    return T, P, imf, t_term, conv


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run(input_path: str, out_root: str | None = None,
        max_iterations: int | None = None, log_level: int | str = "INFO", log_file: str | None = None) -> Dict[str, Any]:
    setup_logging(level=log_level, log_file=log_file)
    log.info("rmgpu run starting – input=%s out_root=%s", input_path, out_root)
    model_input = load_input(input_path)
    log.info("Input loaded")
    databases = _build_databases(model_input)
    log.info("Databases built")
    ml = _build_ml()
    log.info("ML estimators ready")
    fam_sel = getattr(model_input, "database", None)
    fam_sel = fam_sel.kinetics_families if fam_sel is not None else "default"
    families = _build_families(fam_sel)
    log.info("Families built")
    seed_species = _build_seed_species(model_input)
    log.info("Seed species built: %d", len(seed_species))
    seed_mech_species, seed_mech_reactions, seed_mech_summaries = _build_seed_mechanisms(
        model_input, databases)
    log.info("Seed mechanisms built: %d species / %d reactions", len(seed_mech_species), len(seed_mech_reactions))
    T, P, imf, t_term, conv = _build_reactor(model_input)
    log.info("Reactor built: T=%.1f K, P=%.3e Pa", T, P)

    # Model-block tolerances
    tol_move = tol_keep = tol_int = None
    max_edge = 100000
    mb = getattr(model_input, "model", None)
    if mb is not None:
        tol_move = float(getattr(mb, "tolerance_move_to_core", 0.01) or 0.01)
        tol_keep = float(getattr(mb, "tolerance_keep_in_edge", 0.001) or 0.0)
        tol_int = float(getattr(mb, "tolerance_interrupt_simulation", 0.1) or 0.1)
        max_edge = int(getattr(mb, "maximum_edge_species", 100000) or 100000)

    db_block = getattr(model_input, "database", None)
    ctx = RunContext(
        databases=databases,
        ml=ml,
        families=families,
        seed_species=seed_species,
        initial_mole_fractions=imf,
        temperature=T,
        pressure=P,
        config=LoopConfig(
            tolerance_move_to_core=tol_move if tol_move is not None else 0.01,
            tolerance_keep_in_edge=tol_keep if tol_keep is not None else 0.0,
            **({"max_iterations": max_iterations} if max_iterations is not None else {}),
        ),
        thermo_libraries=list(db_block.thermo_libraries or []) if db_block else None,
        reaction_libraries=list(db_block.reaction_libraries or []) if db_block else None,
        termination_time=t_term,
        termination_conversion=conv,
        # job-07/step-06 (pdep): the YAML pressure_dependence block drives the
        # pdep driver (method, T/P grid, interpolation model). When absent,
        # the loop's pdep hook is a no-op (HPL rates stand in).
        pressure_dependence=getattr(model_input, "pressure_dependence", None),
    )
    # Attach seed mechanism data to context via extra attributes
    # (RunContext is a dataclass; we extend it dynamically for job-06 seed support)
    ctx.seed_mechanisms_species = seed_mech_species
    ctx.seed_mechanisms_reactions = seed_mech_reactions
    ctx.seed_mechanisms_summaries = seed_mech_summaries

    log.info("Starting CoreEdgeLoop")
    loop = CoreEdgeLoop(ctx)
    log.info("CoreEdgeLoop instantiated")
    result = loop.run()
    log.info("CoreEdgeLoop finished – iterations=%d", result.iterations)

    # ---- output tree -----------------------------------------------------
    root = out_root or os.path.join(os.path.dirname(os.path.abspath(input_path)),
                                    "run_output")
    _write_output_tree(root, input_path, model_input, result,
                       blocked_ml=ml.blocked, families=families,
                       reactor_name="reactor", temperature=T, pressure=P,
                       loop=loop, seed_summaries=seed_mech_summaries)

    summary = {
        "iterations": result.iterations,
        "core_species_count": len(result.core_keys),
        "core_reaction_count": len(result.core_model.core.reactions),
        "edge_species_count": len(result.edge_keys),
        "edge_reaction_count": len(result.core_model.edge.reactions),
        "core_species_labels": sorted(sp.label for sp in
                                      result.core_model.core.species),
        "seed_mechanisms": [
            {
                "requested": s.get("requested"),
                "resolved_library_name": s.get("resolved_library_name"),
                "resolution": s.get("resolution"),
                "n_species_loaded": s.get("n_species"),
                "n_reactions_loaded": s.get("n_reactions"),
                "n_reactions_dropped_missing_species":
                    s.get("n_reactions_dropped_missing_species", 0),
                "n_reactions_dropped_no_rate":
                    s.get("n_reactions_dropped_no_rate", 0),
                "n_reactions_dropped_multiband":
                    s.get("n_reactions_dropped_multiband", 0),
                "n_thermo_hits": s.get("n_thermo_hits", 0),
            }
            for s in seed_mech_summaries
        ],
        "seed_reaction_count": len(seed_mech_reactions),
        "seed_species_count": len(seed_mech_species),
        "estimation_counts": result.counts.as_dict(),
        "coverage": result.coverage,
        "blocked_ml": ml.blocked,
        "blocked_families": sorted(families.blocked),
        "steady_state": "steady state" in " ".join(result.log_lines),
        "input_path": input_path,
        "out_root": root,
    }
    _print_summary(summary)
    return summary


def _print_summary(summary: Dict[str, Any]) -> None:
    log.info("Iterations: %d (steady state: %s)", summary['iterations'], summary['steady_state'])
    log.info("Core species: %d, Core reactions: %d", summary['core_species_count'], summary['core_reaction_count'])
    log.info("Edge species: %d, Edge reactions: %d", summary['edge_species_count'], summary['edge_reaction_count'])
    for s in summary.get("seed_mechanisms", []):
        log.info("Seed mechanism %r: resolved to %r (%s), %d species / %d reactions loaded (dropped: missing-species %d, no-rate %d, multiband %d)",
                 s['requested'], s['resolved_library_name'], s['resolution'],
                 s['n_species_loaded'], s['n_reactions_loaded'],
                 s['n_reactions_dropped_missing_species'],
                 s['n_reactions_dropped_no_rate'],
                 s['n_reactions_dropped_multiband'])
    log.info("Estimation: %s", summary['estimation_counts'])
    log.info("Coverage gaps: %s", summary['coverage'])


# ---------------------------------------------------------------------------
# Output tree (PLAN 12.3)
# ---------------------------------------------------------------------------

def _checkpoint_hashes():
    from pathlib import Path
    import hashlib
    from rmgpu.ml.base import MODELS_DIR
    out = {}
    for name in ("chemeleon_thermo_662946.ckpt",
                 "chemprop_kinetics_662946.ckpt"):
        p = Path(MODELS_DIR) / name
        if p.exists():
            out[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _git_hash():
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.getcwd(),
            stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _rmgdb_hash():
    from pathlib import Path
    import hashlib
    h = hashlib.sha256()
    for name in ("thermo.db", "kinetics.db"):
        p = Path("/home/jackson/rmgpu/rmgdb/db") / name
        if p.exists():
            with open(p, "rb") as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
    return h.hexdigest()


def _write_output_tree(root: str, input_path: str, model_input: Input,
                       result, blocked_ml, families, reactor_name,
                       temperature: float, pressure: float, loop=None,
                       seed_summaries=None) -> None:
    import json
    from datetime import datetime, timezone
    from rmgpu.schemas.mechanism import (MechanismArtifact, MechanismCore,
                                         SpeciesEntry, ReactionEntry,
                                         ThermoModel, RateParams,
                                         dump_mechanism)

    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "mechanism"), exist_ok=True)
    os.makedirs(os.path.join(root, "species"), exist_ok=True)
    os.makedirs(os.path.join(root, "reactions"), exist_ok=True)
    os.makedirs(os.path.join(root, "profiles", reactor_name), exist_ok=True)

    # run.yaml: the exact resolved input (dump the resolved document - pure
    # YAML-serializable - not the pydantic model whose Quantity fields need a
    # custom representer)
    resolved = load_input(input_path)
    from rmgpu.schemas.input import resolve_extends as _res
    import os as _os
    _base = _os.path.dirname(_os.path.abspath(input_path))
    with open(input_path) as _f:
        _doc = yaml.safe_load(_f)
    _doc = _res(_doc, _base)
    yaml.safe_dump(_doc, open(os.path.join(root, "run.yaml"), "w"))

    # provenance.yaml
    prov = {
        "rmgpu_git": _git_hash(),
        "rmgdb_hash": _rmgdb_hash(),
        "ml_checkpoint_hashes": _checkpoint_hashes(),
        "solver": "torchdae",
        "units": "SI",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": sys.version.split()[0],
            "torch": _pkg_version("torch"),
            "rdkit": _pkg_version("rdkit"),
        },
        "input_path": os.path.abspath(input_path),
        "blocked_ml": list(blocked_ml or []),
        "blocked_families": sorted(getattr(families, "blocked", {})),
        "seed_mechanisms": [
            {
                "requested": s.get("requested"),
                "resolved_library_name": s.get("resolved_library_name"),
                "resolution": s.get("resolution"),
                "species_loaded": s.get("n_species"),
                "reactions_loaded": s.get("n_reactions"),
                "reactions_dropped_missing_species":
                    s.get("n_reactions_dropped_missing_species", 0),
                "reactions_dropped_no_rate": s.get("n_reactions_dropped_no_rate", 0),
                "reactions_dropped_multiband":
                    s.get("n_reactions_dropped_multiband", 0),
                "thermo_hits": s.get("n_thermo_hits", 0),
            }
            for s in (seed_summaries or [])
        ],
    }
    with open(os.path.join(root, "provenance.yaml"), "w") as f:
        yaml.safe_dump(prov, f)

    cm = result.core_model
    # mechanism/core.yaml + edge.yaml
    core_species = [_species_entry(s) for s in cm.core.species]
    core_rxns = [_reaction_entry(r) for r in cm.core.reactions]
    core_core = MechanismCore(rmgpu="1.0", species=core_species, reactions=core_rxns)
    edge_species = [_species_entry(s) for s in cm.edge.species]
    edge_rxns = [_reaction_entry(r) for r in cm.edge.reactions]
    edge_core = MechanismCore(rmgpu="1.0", species=edge_species, reactions=edge_rxns)
    artifact = MechanismArtifact(rmgpu="1.0", core=core_core, edge=edge_core,
                                 provenance={"iteration": result.iterations})
    dump_mechanism(artifact, os.path.join(root, "mechanism", "core.yaml"))
    # edge.yaml: the EDGE model as its own artifact (core field = the edge set)
    artifact_e = MechanismArtifact(rmgpu="1.0", core=edge_core,
                                   provenance={"iteration": result.iterations})
    dump_mechanism(artifact_e, os.path.join(root, "mechanism", "edge.yaml"))

    # PLAN 12.3 legacy interop exports (derived from the canonical artifact)
    try:
        from rmgpu.io.chemkin import write_chemkin
        write_chemkin(artifact, os.path.join(root, "mechanism"))
    except Exception as e:  # noqa: BLE001
        log.warning("chemkin export failed: %s", e)

    # species/*.json
    for s in cm.core.species:
        with open(os.path.join(root, "species", f"{s.label}.json"), "w") as f:
            json.dump(_species_entry(s).model_dump(), f, indent=2)

    # reactions/reactions.json
    with open(os.path.join(root, "reactions", "reactions.json"), "w") as f:
        json.dump([_reaction_entry(r).model_dump() for r in cm.core.reactions],
                  f, indent=2)

    # profiles/<reactor>/time_series.csv + metadata.yaml
    prof = result.final_profiles
    if prof is not None:
        import csv
        # use the species keys (canonical SMILES) of the last simulation
        keys = loop._last_sim["keys"] if loop._last_sim else _sim_species_keys(cm.core.species)
        header = ["time[s]"] + [f"{k}_molefrac" for k in keys]
        with open(os.path.join(root, "profiles", reactor_name,
                               "time_series.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            n = len(prof.times)
            for i in range(n):
                row = [prof.times[i]] + [prof.ys[i][j] for j in range(len(prof.ys[i]))]
                w.writerow(row)
        meta = {
            "units": "mole_fraction",
            "reactor": reactor_name,
            "temperature_K": temperature,
            "pressure_Pa": pressure,
            "t_end_s": prof.times[-1],
            "solver": "torchdae tr_bdf2",
        }
        with open(os.path.join(root, "profiles", reactor_name,
                               "metadata.yaml"), "w") as f:
            yaml.safe_dump(meta, f)

    # summary.md
    counts = result.counts.as_dict()
    lines = [
        "# Summary",
        "",
        f"Core species: {len(cm.core.species)}",
        f"Core reactions: {len(cm.core.reactions)}",
        f"Edge species: {len(cm.edge.species)}",
        f"Edge reactions: {len(cm.edge.reactions)}",
        f"Iterations: {result.iterations}",
        "",
        "Estimation coverage:",
        f"- library_hits: {counts['library_hits']}",
        f"- ml_hits: {counts['ml_hits']}",
        f"- coverage_errors: {counts['coverage_errors']}",
        "",
        "Coverage gaps:",
    ]
    for k, v in result.coverage.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    for ln in result.log_lines:
        lines.append(ln)
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write("\n".join(lines))

    # rmgpu.log
    with open(os.path.join(root, "rmgpu.log"), "w") as f:
        for ln in result.log_lines:
            f.write(ln + "\n")

    # events.jsonl
    with open(os.path.join(root, "events.jsonl"), "w") as f:
        for ev in result.events:
            f.write(json.dumps(ev) + "\n")


def _sim_species_keys(species):
    # deterministic key set for the profile header
    return [s.label for s in species]


def _species_entry(s):
    from rmgpu.schemas.mechanism import SpeciesEntry, ThermoModel
    thermo = s.thermo
    tmodel = ThermoModel(model="ml", Hf298=None, S298=None, Cp=None,
                         method="ml")
    if thermo is not None:
        try:
            tmodel.Hf298 = float(thermo.Hf298)
            tmodel.S298 = float(thermo.S298)
        except Exception:  # noqa: BLE001
            pass
    return SpeciesEntry(
        label=s.label,
        formula=s.molecule.get_formula(),
        smiles=s.molecule.to_smiles(),
        adjlist=s.molecule.to_adjlist() if hasattr(s.molecule, "to_adjlist") else None,
        source="ml" if thermo is not None else "ml",
        symmetry=None,
        thermo=tmodel,
    )


def _reaction_entry(r):
    from rmgpu.schemas.mechanism import ReactionEntry, RateParams
    rp = r.rate_model
    params = {}
    rtype = "Arrhenius"
    if rp is not None:
        params = {"A": float(rp.A), "n": float(rp.n), "Ea": float(rp.Ea),
                  "T0": float(rp.T0), "dS": float(rp.dS),
                  "dH": float(getattr(rp, "dH", 0.0))}
        rtype = "Arrhenius"
    rate = RateParams(type=rtype, params=params,
                      method="ml" if rp is None or rp.source == "ml" else "library")
    return ReactionEntry(
        label=f"rxn_{'_'.join(_slug(x.label) for x in r.reactants)}_to_{'_'.join(_slug(x.label) for x in r.products)}",
        reactants=[x.label for x in r.reactants],
        products=[x.label for x in r.products],
        family=r.family,
        template=None,
        source="ml" if rp is None or rp.source == "ml" else "library",
        degeneracy=float(r.degeneracy),
        rate=rate,
    )


def _slug(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in label)[:24]


def _pkg_version(name: str) -> str:
    import importlib.metadata as meta
    try:
        return meta.version(name)
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        log.error("Usage: python -m rmgpu.main <input.yaml>")
        sys.exit(1)
    run(sys.argv[1])
