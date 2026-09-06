"""rmgdb-backed seed-mechanism loader for job-06.

Resolves a mechanism name to a kinetics library id in rmgdb/db/kinetics.db,
then loads species and reactions from the library tables. The public API is
``load_seed_mechanism(name, databases)`` which returns
``(species_list, reaction_list, summary)`` with rmgpu Species and Reaction
objects and a small loud summary of what was loaded and what was dropped.

Honesty rules (job-06/step-07):
  * The name resolution is loud: it tries the exact name, then a small alias
    table (RMG-Py library names differ from rmgdb names, e.g. GRI-Mech3.0-N ->
    GRI-Mech3), then a controlled normalization. Which rule resolved is
    recorded in the summary. If nothing resolves it RAISES (the caller must
    not swallow it).
  * No placeholder species. A reaction whose species label is not in the
    library dictionary is SKIPPED and counted as a ``seed_species_gap``
    (never a silent methane molecule).
  * Thermo is attached (SI) when the run's thermo libraries cover the label;
    otherwise it is left None for the loop's ML estimation (library-hit-first).
  * Multi-band (multi-row) Arrhenius reactions CANNOT be flattened into a
    single A/n/Ea/T0 RateParam (the artifact schema is flat-only). They are
    dropped and counted as a recorded gap - the pdep machinery (job-07) is
    where banded/falloff models get first-class treatment.
  * A-factor and Ea are converted by the ACTUAL stored unit (job-02's
    ``convert_A`` / ``convert_Ea``), so ``cm^6/(mol^2*s)`` termolecular rows get
    the 1e-12 factor (not the old 1e-6 substring heuristic) and ``cal/mol`` Ea
    gets the 4.184 factor.

The loaded reactions carry a flat :class:`rmgpu.reactor.simulator.RateParam`
(the same type the core/edge loop uses for generated reactions), so the
mechanism artifact writer (``rmgpu.main._reaction_entry``) can serialize them.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

from rmgpu.core.loop import canonical_key
from rmgpu.core.model import Reaction, Species
from rmgpu.data.estimation import (
    MLCoverageError,
    ThermoPrediction,
    _library_H298_S298,
    _library_cp_model,
)
from rmgpu.data.kinetics import convert_A, convert_Ea
from rmgpu.molecule.molecule import Molecule
from rmgpu.reactor.simulator import RateParam
from rmgpu.db import Databases
from rmgpu.logging import get_logger

log = get_logger("db.seed_loader")

# ---------------------------------------------------------------------------
# Seed-mechanism name resolution (RMG-Py library names vs rmgdb library names)
# ---------------------------------------------------------------------------
SEED_LIBRARY_ALIASES: dict[str, str] = {
    "GRI-Mech3.0-N": "GRI-Mech3",
    "GRI-Mech3.0": "GRI-Mech3",
    "GRI-Mech 3.0": "GRI-Mech3",
}

def _normalize_library_name(name: str) -> str:
    n = name.strip()
    n = re.sub(r"\.0+(-N?)?$", "", n)
    n = re.sub(r"-N$", "", n)
    return n

def _library_id_by_name(conn: sqlite3.Connection, name: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT id, name FROM kinetics_libraries_table WHERE name = ?", (name,)
    ).fetchone()
    if row:
        return int(row[0]), row[1], "exact"
    real = SEED_LIBRARY_ALIASES.get(name)
    if real is not None:
        row = cur.execute(
            "SELECT id, name FROM kinetics_libraries_table WHERE name = ?", (real,)
        ).fetchone()
        if row:
            return int(row[0]), row[1], f"alias:{name}->{real}"
    norm = _normalize_library_name(name)
    if norm and norm != name:
        row = cur.execute(
            "SELECT id, name FROM kinetics_libraries_table WHERE name = ?", (norm,)
        ).fetchone()
        if row:
            return int(row[0]), row[1], f"normalize:{name}->{norm}"
    return None, None, None

def _load_library_species(conn: sqlite3.Connection, library_id: int) -> List[dict]:
    cur = conn.cursor()
    cur.execute(
        "SELECT label, adjacency_list FROM kinetics_library_dictionary_table WHERE library_id = ?",
        (library_id,),
    )
    rows = cur.fetchall()
    return [{"label": r[0], "adjacency_list": r[1]} for r in rows]

def _load_library_reactions(conn: sqlite3.Connection, library_id: int) -> List[dict]:
    cur = conn.cursor()
    cur.execute(
        "SELECT id, label, degeneracy, reversible FROM kinetics_library_reactions_table WHERE library_id = ?",
        (library_id,),
    )
    rows = cur.fetchall()
    return [
        {"id": int(r[0]), "label": r[1],
         "degeneracy": float(r[2] or 0.0), "reversible": r[3]}
        for r in rows
    ]

def _load_all_reaction_species(conn: sqlite3.Connection, library_id: int) -> dict[int, List[dict]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT r.id, s.species_label, s.role
        FROM kinetics_library_reactions_table r
        JOIN kinetics_library_reaction_species_table s ON s.library_reaction_id = r.id
        WHERE r.library_id = ?
        """,
        (library_id,),
    )
    rows = cur.fetchall()
    out: dict[int, List[dict]] = {}
    for rid, label, role in rows:
        out.setdefault(rid, []).append({"label": label, "role": role})
    return out

def _load_all_arrhenius(conn: sqlite3.Connection, library_id: int) -> dict[int, List[dict]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT r.id, a.A_val, a.A_unit, a.n, a.Ea_val, a.Ea_unit, a.T0_val
        FROM kinetics_library_reactions_table r
        JOIN kinetics_arrhenius_table a ON a.library_reaction_id = r.id
        WHERE r.library_id = ?
        ORDER BY r.id, a.Tmin_val, a.Tmax_val
        """,
        (library_id,),
    )
    cols = [d[0] for d in cur.description]
    out: dict[int, List[dict]] = {}
    for r in cur.fetchall():
        rid = int(r[0])
        row = dict(zip(cols, r))
        out.setdefault(rid, []).append(row)
    return out

def _flat_rate_param(rows: List[dict], reversible: bool, degeneracy: float) -> RateParam:
    row = rows[0]
    A = convert_A(float(row["A_val"] or 0.0), row.get("A_unit"))
    Ea = convert_Ea(float(row["Ea_val"] or 0.0), row.get("Ea_unit"))
    n = float(row.get("n") or 0.0)
    T0 = float(row.get("T0_val") or 1.0)
    return RateParam(A=A, n=n, Ea=Ea, T0=T0, dS=0.0, reversible=reversible,
                     family="seed", template=None,
                     degeneracy=float(degeneracy or 1.0), source="library")

def _resolve_library_thermo(databases: Databases, label: str) -> Optional[ThermoPrediction]:
    entry = None
    libs = list(getattr(databases, "thermo_libraries", None) or [])
    for lib in libs:
        try:
            e = databases.thermo.get_entry_grouped_by_label(label, lib)
        except Exception:
            e = None
        if e is not None:
            entry = e
            break
    if entry is None and not libs:
        try:
            entry = databases.thermo.get_entry_grouped_by_label(label)
        except Exception:
            entry = None
    if entry is None:
        return None
    try:
        H298, S298 = _library_H298_S298(entry)
        cp_model = _library_cp_model(entry)
    except MLCoverageError:
        return None
    return ThermoPrediction(
        Hf298=H298, S298=S298, Cp_model=cp_model,
        uncertainties={"source": "library", "label": label},
    )

def load_seed_mechanism(name: str, databases: Databases) -> Tuple[List[Species], List[Reaction], dict]:
    log.info("seed_loader: opening kinetics DB")
    kinetics_path = Path(databases.kinetics.db_path)
    conn = sqlite3.connect(str(kinetics_path))
    try:
        log.debug("seed_loader: resolving library name %r", name)
        lib_id, resolved_name, how = _library_id_by_name(conn, name)
        if lib_id is None:
            raise ValueError(
                f"Seed mechanism {name!r} not found in rmgdb "
                "(tried exact name, alias table, and normalization)"
            )
        summary = {
            "requested_name": name,
            "resolved_library_name": resolved_name,
            "resolution": how,
            "n_species": 0,
            "n_reactions": 0,
            "n_species_dropped_malformed": 0,
            "n_reactions_dropped_missing_species": 0,
            "n_reactions_dropped_no_rate": 0,
            "n_reactions_dropped_multiband": 0,
            "n_thermo_hits": 0,
        }
        log.info("seed_loader: resolved %r -> %r via %s (id=%s)", name, resolved_name, how, lib_id)
        log.debug("seed_loader: loading species for library_id=%s", lib_id)
        lib_species_rows = _load_library_species(conn, lib_id)
        log.info("seed_loader: loaded %d species rows", len(lib_species_rows))
        species_map: dict[str, Species] = {}
        species_list: List[Species] = []
        label_to_species: dict[str, Species] = {}
        for row in lib_species_rows:
            label = row["label"]
            adj = row["adjacency_list"]
            try:
                mol = Molecule.from_adjacency_list(adj)
            except Exception:
                summary["n_species_dropped_malformed"] += 1
                continue
            key = canonical_key(mol)
            sp = species_map.get(key)
            if sp is None:
                thermo = _resolve_library_thermo(databases, label)
                sp = Species(label=label, molecule=mol, reactive=True, thermo=thermo)
                species_map[key] = sp
                species_list.append(sp)
                if thermo is not None:
                    summary["n_thermo_hits"] += 1
            label_to_species[label] = sp
        summary["n_species"] = len(species_list)

        log.debug("seed_loader: loading reactions for library_id=%s", lib_id)
        lib_reactions = _load_library_reactions(conn, lib_id)
        log.info("seed_loader: loaded %d reaction rows", len(lib_reactions))
        species_by_rxn = _load_all_reaction_species(conn, lib_id)
        arrhenius_by_rxn = _load_all_arrhenius(conn, lib_id)
        reaction_list: List[Reaction] = []
        for rrow in lib_reactions:
            reaction_id = rrow["id"]
            species_rows = species_by_rxn.get(reaction_id, [])
            reactants: List[Species] = []
            products: List[Species] = []
            missing = False
            for srow in species_rows:
                lbl = srow["label"]
                sp = label_to_species.get(lbl)
                if sp is None:
                    missing = True
                    break
                if srow["role"] == "reactant":
                    reactants.append(sp)
                else:
                    products.append(sp)
            if missing:
                summary["n_reactions_dropped_missing_species"] += 1
                continue
            if not reactants or not products:
                summary["n_reactions_dropped_no_rate"] += 1
                continue
            rows = arrhenius_by_rxn.get(reaction_id, [])
            if not rows:
                summary["n_reactions_dropped_no_rate"] += 1
                continue
            if len(rows) > 1:
                summary["n_reactions_dropped_multiband"] += 1
                continue
            reversible = True if rrow["reversible"] is None else bool(rrow["reversible"])
            rp = _flat_rate_param(rows, reversible, rrow["degeneracy"])
            hf = lambda s: float(s.thermo.Hf298) if s.thermo is not None else 0.0
            s298 = lambda s: float(s.thermo.S298) if s.thermo is not None else 0.0
            rp.dH = sum(hf(p) for p in products) - sum(hf(r) for r in reactants)
            rp.dS = sum(s298(p) for p in products) - sum(s298(r) for r in reactants)
            rxn = Reaction(
                reactants=reactants, products=products, rate_model=rp,
                degeneracy=float(rrow["degeneracy"] or 1.0),
                reversible=rp.reversible, family="seed",
            )
            reaction_list.append(rxn)
        summary["n_reactions"] = len(reaction_list)
    finally:
        log.info("seed_loader: closing DB connection")
        conn.close()
    log.info("seed_loader: done – %d species, %d reactions", len(species_list), len(reaction_list))
    return species_list, reaction_list, summary
