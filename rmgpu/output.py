"""Output tree writer per PLAN.md 12.3.

Extensible registry of per-dir writers. The core writer creates:
run/
  run.yaml
  provenance.yaml
  summary.md
  rmgpu.log
  events.jsonl
  mechanism/core.yaml
  mechanism/edge.yaml
  species/*.json + *.svg
  reactions/reactions.json
  profiles/<reactor>/time_series.csv + metadata.yaml
"""

from __future__ import annotations

import os
import json
import yaml
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List

from rmgpu.schemas.mechanism import MechanismArtifact, MechanismCore, SpeciesEntry, ReactionEntry, ThermoModel


def _git_hash() -> str:
    import subprocess
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd(), stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _versions() -> Dict[str, str]:
    import importlib.metadata as meta
    versions = {}
    for pkg in ["torch", "rdkit", "pint", "cantera"]:
        try:
            versions[pkg] = meta.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    import sys
    versions["python"] = sys.version.split()[0]
    return versions


def write_provenance(root: str, input_path: str, extra: Dict[str, Any] = None) -> None:
    prov = {
        "rmgpu_git": _git_hash(),
        "rmgdb_version": "unknown",
        "rmgdb_hash": "unknown",
        "ml_checkpoint_hashes": {},
        "solver": "torchdae",
        "units": "SI",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "versions": _versions(),
        "input_path": input_path,
    }
    if extra:
        prov.update(extra)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "provenance.yaml"), "w") as f:
        yaml.safe_dump(prov, f)


def write_run_yaml(root: str, input_path: str) -> None:
    # Copy the resolved input verbatim
    import yaml
    with open(input_path) as f:
        data = yaml.safe_load(f)
    with open(os.path.join(root, "run.yaml"), "w") as f:
        yaml.safe_dump(data, f)


def write_summary(root: str, core_count: int, edge_count: int, iterations: int, estimation_counts: Dict[str, Any]) -> None:
    lines = [
        "# Summary",
        "",
        f"Core species: {core_count}",
        f"Edge species: {edge_count}",
        f"Iterations: {iterations}",
        "",
        "Estimation coverage:",
    ]
    for k, v in estimation_counts.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Provenance written to provenance.yaml")
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write("\n".join(lines))


def write_log(root: str, log_text: str) -> None:
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "rmgpu.log"), "w") as f:
        f.write(log_text)


def write_events(root: str, events: List[Dict[str, Any]]) -> None:
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "events.jsonl"), "w") as f:
        for ev in (events or []):
            f.write(json.dumps(ev) + "\n")


def _species_to_entry(sp) -> SpeciesEntry:
    label = sp.label
    formula = getattr(sp.molecule, "formula", None) if getattr(sp, "molecule", None) else None
    smiles = getattr(sp.molecule, "smiles", None) if getattr(sp, "molecule", None) else None
    adjlist = getattr(sp.molecule, "adjlist", None) if getattr(sp, "molecule", None) else None
    thermo = sp.thermo
    thermo_model = ThermoModel(model="ml", Hf298=None, S298=None, Cp=None, method="ml", uncertainty=None)
    if thermo is not None:
        # Best-effort extraction
        try:
            thermo_model.Hf298 = float(getattr(thermo, "Hf298", 0.0))
            thermo_model.S298 = float(getattr(thermo, "S298", 0.0))
        except Exception:
            pass
    return SpeciesEntry(
        label=label,
        formula=formula,
        smiles=smiles,
        adjlist=adjlist,
        source="ml",
        symmetry=None,
        thermo=thermo_model,
    )


def _reaction_to_entry(rx) -> ReactionEntry:
    from rmgpu.core.model import Species as CoreSpecies
    reactants = [s.label for s in rx.reactants]
    products = [s.label for s in rx.products]
    from rmgpu.schemas.mechanism import RateParams
    rate = RateParams(type="Arrhenius", params={}, method="ml", uncertainty=None)
    return ReactionEntry(
        label=f"rxn_{hash(tuple(reactants + products)) & 0xffffffff}",
        reactants=reactants,
        products=products,
        family=getattr(rx, "family", None),
        template=None,
        source="ml",
        degeneracy=float(getattr(rx, "degeneracy", 1.0)),
        rate=rate,
    )


def write_mechanism(root: str, core_model, estimation_counts: Dict[str, Any] = None) -> MechanismArtifact:
    mech_dir = os.path.join(root, "mechanism")
    os.makedirs(mech_dir, exist_ok=True)

    core_species = [_species_to_entry(s) for s in getattr(core_model.core, "species", [])]
    core_reactions = [_reaction_to_entry(r) for r in getattr(core_model.core, "reactions", [])]

    core_core = MechanismCore(
        rmgpu="1.0",
        species=core_species,
        reactions=core_reactions,
    )
    artifact = MechanismArtifact(rmgpu="1.0", core=core_core, provenance={"iteration": getattr(core_model, "iteration_num", 0)})

    path_core = os.path.join(mech_dir, "core.yaml")
    from rmgpu.schemas.mechanism import dump_mechanism
    dump_mechanism(artifact, path_core)

    # Edge placeholder
    edge_core = MechanismCore(rmgpu="1.0", species=[], reactions=[])
    artifact_edge = MechanismArtifact(rmgpu="1.0", core=edge_core)
    dump_mechanism(artifact_edge, os.path.join(mech_dir, "edge.yaml"))

    # Species files
    species_dir = os.path.join(root, "species")
    os.makedirs(species_dir, exist_ok=True)
    for s in getattr(core_model.core, "species", []):
        entry = _species_to_entry(s)
        with open(os.path.join(species_dir, f"{entry.label}.json"), "w") as f:
            json.dump(entry.model_dump(), f, indent=2)

    # Reactions file
    reactions_dir = os.path.join(root, "reactions")
    os.makedirs(reactions_dir, exist_ok=True)
    with open(os.path.join(reactions_dir, "reactions.json"), "w") as f:
        json.dump([r.model_dump() for r in core_reactions], f, indent=2)

    return artifact


def write_profiles(root: str, reactor_name: str, times: List[float], species_data: Dict[str, List[float]]) -> None:
    profiles_dir = os.path.join(root, "profiles", reactor_name)
    os.makedirs(profiles_dir, exist_ok=True)
    import csv
    header = ["time[s]"] + [f"{k}_molefrac" for k in species_data.keys()]
    with open(os.path.join(profiles_dir, "time_series.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, t in enumerate(times):
            row = [t] + [species_data[k][i] for k in species_data]
            writer.writerow(row)
    # metadata
    meta = {"units": "si", "reactor": reactor_name}
    with open(os.path.join(profiles_dir, "metadata.yaml"), "w") as f:
        yaml.safe_dump(meta, f)


def write_output_tree(root: str, input_path: str, core_model, log_text: str = "", events: List[Dict[str, Any]] = None, estimation_counts: Dict[str, Any] = None) -> None:
    """Write full output tree per PLAN.md 12.3."""
    os.makedirs(root, exist_ok=True)
    write_provenance(root, input_path)
    write_run_yaml(root, input_path)
    write_log(root, log_text)
    # events.jsonl always exists, even if empty
    write_events(root, events or [])
    core_count = len(getattr(core_model.core, "species", []))
    edge_count = len(getattr(core_model.edge, "species", []))
    iterations = getattr(core_model, "iteration_num", 0)
    write_summary(root, core_count, edge_count, iterations, estimation_counts or {})
    write_mechanism(root, core_model, estimation_counts)
    # Profiles placeholder
    profiles_dir = os.path.join(root, "profiles")
    os.makedirs(profiles_dir, exist_ok=True)
