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


# ---------------------------------------------------------------------------
# Seed-mechanism name resolution (RMG-Py library names vs rmgdb library names)
# ---------------------------------------------------------------------------
# RMG-Py ships mechanisms under names that do not always match the rmgdb
# kinetics_library name. Keep this table explicit and loud; the controlled
# normalization below is a second safety net.
SEED_LIBRARY_ALIASES: dict[str, str] = {
    "GRI-Mech3.0-N": "GRI-Mech3",
    "GRI-Mech3.0": "GRI-Mech3",
    "GRI-Mech 3.0": "GRI-Mech3",
}


def _normalize_library_name(name: str) -> str:
    """Strip a trailing version token (e.g. 'GRI-Mech3.0' -> 'GRI-Mech3',
    'GRI-Mech3.0-N' -> 'GRI-Mech3')."""
    n = name.strip()
    n = re.sub(r"\.0+(-N?)?$", "", n)
    n = re.sub(r"-N$", "", n)
    return n


def _library_id_by_name(conn: sqlite3.Connection, name: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Resolve a mechanism name to a kinetics library id.

    Returns (library_id, resolved_name, how). ``how`` is one of 'exact',
    'alias:<orig>-><real>', 'normalize:<orig>-><norm>'. (None, None, None) when
    nothing resolves (the caller raises)."""
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


def _load_reaction_species(conn: sqlite3.Connection, reaction_id: int) -> List[dict]:
    cur = conn.cursor()
    cur.execute(
        "SELECT species_label, role FROM kinetics_library_reaction_species_table WHERE library_reaction_id = ?",
        (reaction_id,),
    )
    rows = cur.fetchall()
    return [{"label": r[0], "role": r[1]} for r in rows]


def _load_arrhenius_rows(conn: sqlite3.Connection, reaction_id: int) -> List[dict]:
    """ALL stored Arrhenius rows for a reaction (multi-band aware)."""
    cur = conn.cursor()
    cur.execute(
        "SELECT A_val, A_unit, n, Ea_val, Ea_unit, T0_val FROM kinetics_arrhenius_table "
        "WHERE library_reaction_id = ? ORDER BY Tmin_val, Tmax_val",
        (reaction_id,),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _flat_rate_param(rows: List[dict], reversible: bool, degeneracy: float) -> RateParam:
    """Build a flat SI RateParam from a SINGLE stored Arrhenius row.

    A and Ea are converted by the ACTUAL stored unit (job-02's conversion
    helpers), so ``cm^6/(mol^2*s)`` gets 1e-12 and ``cal/mol`` gets 4.184.
    ``dS`` is 0.0 here (filled in from participant thermo by the caller).
    """
    row = rows[0]
    A = convert_A(float(row["A_val"] or 0.0), row.get("A_unit"))
    Ea = convert_Ea(float(row["Ea_val"] or 0.0), row.get("Ea_unit"))
    n = float(row.get("n") or 0.0)
    T0 = float(row.get("T0_val") or 1.0)
    return RateParam(A=A, n=n, Ea=Ea, T0=T0, dS=0.0, reversible=reversible,
                     family="seed", template=None,
                     degeneracy=float(degeneracy or 1.0), source="library")


def _resolve_library_thermo(databases: Databases, label: str) -> Optional[ThermoPrediction]:
    """Resolve an SI ThermoPrediction for ``label`` from the run's thermo
    libraries (library-hit-first). None when no library covers the label (or
    the row has no usable model) - the loop's ML estimation then handles it.
    Reuses the run's ThermoDB facade (no per-label reload)."""
    entry = None
    libs = list(getattr(databases, "thermo_libraries", None) or [])
    for lib in libs:
        try:
            e = databases.thermo.get_entry_grouped_by_label(label, lib)
        except Exception:  # noqa: BLE001
            e = None
        if e is not None:
            entry = e
            break
    if entry is None and not libs:
        try:
            entry = databases.thermo.get_entry_grouped_by_label(label)
        except Exception:  # noqa: BLE001
            entry = None
    if entry is None:
        return None
    try:
        H298, S298 = _library_H298_S298(entry)
        cp_model = _library_cp_model(entry)
    except MLCoverageError:
        # Row exists but carries no usable model: leave thermo None (loop's ML
        # branch decides) - never a silent wrong value.
        return None
    return ThermoPrediction(
        Hf298=H298, S298=S298, Cp_model=cp_model,
        uncertainties={"source": "library", "label": label},
    )
def load_seed_mechanism(name: str, databases: Databases) -> Tuple[List[Species], List[Reaction], dict]:
    """Load a seed mechanism from rmgdb.

    Args:
        name: mechanism name from the input, e.g. 'GRI-Mech3.0-N'.
        databases: rmgpu.db.Databases instance (paths + thermo scope are used).

    Returns:
        (species_list, reaction_list, summary). Species are deduped by
        canonical SMILES and carry SI library thermo when available. Reactions
        carry a flat SI RateParam. ``summary`` is a loud record of the
        resolution and every drop (see the module docstring).
    """
    kinetics_path = Path(databases.kinetics.db_path)
    conn = sqlite3.connect(str(kinetics_path))
    try:
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
        # -- species ---------------------------------------------------------
        lib_species_rows = _load_library_species(conn, lib_id)
        species_map: dict[str, Species] = {}
        species_list: List[Species] = []
        label_to_species: dict[str, Species] = {}
        for row in lib_species_rows:
            label = row["label"]
            adj = row["adjacency_list"]
            try:
                mol = Molecule.from_adjacency_list(adj)
            except Exception:  # noqa: BLE001
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
            # every dictionary label maps to a (possibly deduped) species, so
            # reaction label lookups never hit "missing" for well-formed libs.
            label_to_species[label] = sp
        summary["n_species"] = len(species_list)

        # -- reactions -------------------------------------------------------
        lib_reactions = _load_library_reactions(conn, lib_id)
        reaction_list: List[Reaction] = []
        for rrow in lib_reactions:
            reaction_id = rrow["id"]
            species_rows = _load_reaction_species(conn, reaction_id)
            reactants: List[Species] = []
            products: List[Species] = []
            missing = False
            for srow in species_rows:
                lbl = srow["label"]
                sp = label_to_species.get(lbl)
                if sp is None:
                    # Species not in the library dictionary: SKIP + count
                    # (loud). Never inject a placeholder molecule.
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
            rows = _load_arrhenius_rows(conn, reaction_id)
            if not rows:
                summary["n_reactions_dropped_no_rate"] += 1
                continue
            if len(rows) > 1:
                # Multi-band / multi-row: cannot flatten to one A/n/Ea/T0.
                # Drop + record (pdep/banded handling is job-07 scope).
                summary["n_reactions_dropped_multiband"] += 1
                continue
            reversible = True if rrow["reversible"] is None else bool(rrow["reversible"])
            rp = _flat_rate_param(rows, reversible, rrow["degeneracy"])
            # dS / dH from participant thermo (best-effort; 0 where missing) so
            # the reverse factor (1/K_c, thermodynamically consistent) and the
            # artifact carry meaningful reaction Gibbs energy. job-06/step-07:
            # the previous entropy-only reverse factor was off by up to 30
            # orders of magnitude for exothermic reactions (root cause of the
            # non-physical c3h4 profile); dH is now required for it.
            hf = lambda s: float(s.thermo.Hf298) if s.thermo is not None else 0.0  # noqa: E731
            s298 = lambda s: float(s.thermo.S298) if s.thermo is not None else 0.0  # noqa: E731
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
        conn.close()
    return species_list, reaction_list, summary
