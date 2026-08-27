"""Database facades over rmgdb SQLite databases."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from rmgpu.data.entries import (
    ThermoEntry,
    KineticsEntry,
    TransportEntry,
    StatMechEntry,
    StatMechGroup,
    SolvationEntry,
    SoluteLibraryEntry,
    SolventLibraryEntry,
)


def _series_to_dict(row: pd.Series) -> dict:
    """Convert a pandas Series (a view row) to a plain dict, NaN -> None.

    The gate normalizer (gates/normalizer.py) treats both ``None`` and NaN as
    "absent", but unit columns and YAML dumping behave most predictably with
    None, so NaN floats are normalized away here.
    """
    out: dict = {}
    for k, v in row.items():
        if isinstance(v, float) and pd.isna(v):
            out[k] = None
        else:
            out[k] = v
    return out


class ThermoDB:
    """Facade for accessing thermo data from rmgdb SQLite database."""

    def __init__(self, db_path: str | Path = "/home/jackson/rmgpu/rmgdb/db/thermo.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self._df: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        """Load the thermo_libraries_view into a DataFrame (cached)."""
        if self._df is None:
            self._df = pd.read_sql(
                "SELECT * FROM thermo_libraries_view",
                f"sqlite:///{self.db_path}",
            ).set_index("id")
        return self._df

    def get_entry_count(self) -> int:
        """Return the total number of thermo library entries."""
        return len(self._load())

    def get_entry_count_by_library(self, library_name: str) -> int:
        """Return the number of thermo entries in a named library.
        
        Uses COUNT(DISTINCT label) because the thermo_libraries_view may have
        duplicate rows for species with multiple data models (Wilhoit + NASA).
        """
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(DISTINCT label) FROM thermo_libraries_view WHERE name = ?",
                (library_name,)
            )
            return cur.fetchone()[0]

    def get_entry_by_label(self, label: str) -> Optional[ThermoEntry]:
        """Look up a thermo entry by its label (first matching row)."""
        df = self._load()
        matches = df[df["label"] == label]
        if matches.empty:
            return None
        return self._row_to_entry(matches.iloc[0])

    def get_entry_grouped_by_label(self, label: str, library: Optional[str] = None) -> Optional[ThermoEntry]:
        """Merge all rows for a label into one ThermoEntry.

        The view stores one row per data model: NASA7 segments (rows with c1
        set) and/or a Cp-data row (rows with Tdata_1 set). This mirrors how
        RMG-Py loads a single library entry carrying both models. Pass
        ``library`` to scope to one library (default: all libraries, first
        row wins for scalar fields).
        """
        df = self._load()
        sel = df["label"] == label
        if library is not None:
            sel = sel & (df["name"] == library)
        matches = df[sel]
        if matches.empty:
            return None

        entry = self._row_to_entry(matches.iloc[0])
        # reset lists so we don't double-count the first row
        entry.nasa_polynomials = []
        entry.Tdata, entry.Cpdata = [], []
        for _, row in matches.iterrows():
            # merge NASA segments
            c1 = row.get("c1")
            if pd.notna(c1):
                entry.nasa_polynomials.append({
                    "coeffs": [float(row.get(f"c{i}", 0) or 0) for i in range(1, 8)],
                    "Tmin": float(row.get("poly_Tmin", 0) or 0),
                    "Tmax": float(row.get("poly_Tmax", 0) or 0),
                })
            # merge Cp data points
            if pd.notna(row.get("Tdata_1")) and not entry.Tdata:
                tdata, cdata = [], []
                for i in range(1, 8):
                    td = row.get(f"Tdata_{i}")
                    cd = row.get(f"Cpdata_{i}")
                    if pd.notna(td):
                        tdata.append(float(td))
                        if pd.notna(cd):
                            cdata.append(float(cd))
                entry.Tdata, entry.Cpdata = tdata, cdata
            if pd.notna(row.get("E0")) and entry.E0 is None:
                entry.E0 = float(row["E0"])
                entry.E0_unit = str(row.get("E0_unit") or "kcal/mol")
        return entry

    def get_entries_by_library(self, library_name: str) -> list[ThermoEntry]:
        """Return all entries from a named library."""
        df = self._load()
        matches = df[df["name"] == library_name]
        return [self._row_to_entry(row) for _, row in matches.iterrows()]

    def is_in_library(self, label: str, library_name: str) -> bool:
        """Return True if a label exists in the given library."""
        df = self._load()
        return len(df[(df["label"] == label) & (df["name"] == library_name)]) > 0

    def get_library_labels(self, library_name: str) -> list[str]:
        """Return the sorted unique labels present in a named library."""
        df = self._load()
        return sorted(df.loc[df["name"] == library_name, "label"].unique().tolist())

    def get_raw_rows_by_library(self, library_name: str) -> dict[str, list[dict]]:
        """Raw thermo_libraries_view rows for a library, grouped by label.

        Returns {label: [view-row-dict, ...]}. The view fans one logical entry
        out to several rows (NASA segments, ThermoData, and the rmgdb
        duplicate-row artifact), so the gate normalizer groups them back by
        label - exactly mirroring RMG-Py's one-entry-per-label structure.
        """
        df = self._load()
        sel = df[df["name"] == library_name]
        out: dict[str, list[dict]] = {}
        for _, row in sel.iterrows():
            out.setdefault(str(row["label"]), []).append(_series_to_dict(row))
        return out

    def get_raw_label_rows(self, label: str, library: Optional[str] = None) -> list[dict]:
        """Raw view rows for a single label. Scope to one library or use all."""
        df = self._load()
        mask = df["label"] == label
        if library is not None:
            mask = mask & (df["name"] == library)
        return [_series_to_dict(r) for _, r in df[mask].iterrows()]

    def _row_to_entry(self, row: pd.Series) -> ThermoEntry:
        """Convert a pandas Series to a ThermoEntry."""
        # Build NASA polynomial list
        nasa_polys = []
        c1_val = row.get("c1")
        if pd.notna(c1_val):
            coeffs = [
                float(c1_val),
                float(row.get("c2", 0) or 0),
                float(row.get("c3", 0) or 0),
                float(row.get("c4", 0) or 0),
                float(row.get("c5", 0) or 0),
                float(row.get("c6", 0) or 0),
                float(row.get("c7", 0) or 0),
            ]
            nasa_polys.append({
                "coeffs": coeffs,
                "Tmin": float(row.get("poly_Tmin", 0) or 0),
                "Tmax": float(row.get("poly_Tmax", 0) or 0),
            })

        # Build Tdata/Cpdata lists from Tdata_1..Tdata_7
        tdata = []
        cdata = []
        for i in range(1, 8):
            td = row.get(f"Tdata_{i}")
            cd = row.get(f"Cpdata_{i}")
            if pd.notna(td):
                tdata.append(float(td))
                if pd.notna(cd):
                    cdata.append(float(cd))

        def _float_or_none(key: str) -> Optional[float]:
            val = row.get(key)
            if pd.isna(val):
                return None
            return float(val)

        def _str_or_default(key: str, default: str) -> str:
            val = row.get(key)
            if pd.isna(val):
                return default
            return str(val)

        return ThermoEntry(
            label=str(row["label"]),
            short_description=_str_or_default("short_description", ""),
            long_description=_str_or_default("long_description", ""),
            Tdata_unit=_str_or_default("Tdata_unit", "K"),
            Cpdata_unit=_str_or_default("Cpdata_unit", "cal/(mol*K)"),
            H298=float(row.get("H298", 0) or 0),
            H298_unit=_str_or_default("H298_unit", "kcal/mol"),
            S298=float(row.get("S298", 0) or 0),
            S298_unit=_str_or_default("S298_unit", "cal/(mol*K)"),
            Tdata=tdata,
            Cpdata=cdata,
            # NOTE: rmgdb does NOT store the Wilhoit fit coefficients (a0..a3,
            # B, H0, S0) or Cp0/CpInf - the view's Cp0/CpInf columns exist but
            # are entirely NULL (verified on the 2026-08-26 rmgdb build).
            # These stay None; see reports/job-02.md gap list.
            nasa_polynomials=nasa_polys,
            nasa_Tmin=_float_or_none("nasa_Tmin"),
            nasa_Tmax=_float_or_none("nasa_Tmax"),
            nasa_T_unit=_str_or_default("nasa_T_unit", "K"),
            E0=_float_or_none("E0"),
            E0_unit=_str_or_default("E0_unit", "kcal/mol"),
        )


class KineticsDB:
    """Facade for accessing kinetics data from rmgdb SQLite database."""

    def __init__(self, db_path: str | Path = "/home/jackson/rmgpu/rmgdb/db/kinetics.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self._df_reactions: Optional[pd.DataFrame] = None
        self._df_families: Optional[pd.DataFrame] = None
        self._df_species: Optional[pd.DataFrame] = None

    def _load_reactions(self) -> pd.DataFrame:
        """Load library reactions into a DataFrame (cached)."""
        if self._df_reactions is None:
            self._df_reactions = pd.read_sql(
                "SELECT * FROM kinetics_library_reactions_table",
                f"sqlite:///{self.db_path}",
            )
        return self._df_reactions

    def _load_families(self) -> pd.DataFrame:
        """Load family definitions into a DataFrame (cached)."""
        if self._df_families is None:
            self._df_families = pd.read_sql(
                "SELECT * FROM kinetics_families_table",
                f"sqlite:///{self.db_path}",
            )
        return self._df_families

    def _load_species(self) -> pd.DataFrame:
        """Load reaction species into a DataFrame (cached)."""
        if self._df_species is None:
            self._df_species = pd.read_sql(
                "SELECT * FROM kinetics_library_reaction_species_table",
                f"sqlite:///{self.db_path}",
            )
        return self._df_species

    def get_library_count(self) -> int:
        """Return the number of kinetics libraries."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM kinetics_libraries_table")
            return cur.fetchone()[0]

    def get_reaction_count(self) -> int:
        """Return the total number of kinetics library reactions."""
        return len(self._load_reactions())

    def get_family_count(self) -> int:
        """Return the total number of reaction families."""
        return len(self._load_families())

    def get_family_by_name(self, name: str) -> Optional[dict]:
        """Look up a family definition by name."""
        df = self._load_families()
        matches = df[df["name"] == name]
        if matches.empty:
            return None
        return matches.iloc[0].to_dict()

    def get_reaction_by_label(self, label: str) -> Optional[dict]:
        """Look up a reaction by label (returns species info too)."""
        df = self._load_reactions()
        matches = df[df["label"] == label]
        if matches.empty:
            return None
        
        reaction = matches.iloc[0].to_dict()
        reaction_id = reaction["id"]
        
        # Get species for this reaction
        species_df = self._load_species()
        species_matches = species_df[species_df["library_reaction_id"] == reaction_id]
        reactants = species_matches[species_matches["role"] == "reactant"]["species_label"].tolist()
        products = species_matches[species_matches["role"] == "product"]["species_label"].tolist()
        
        reaction["reactants"] = reactants
        reaction["products"] = products
        return reaction
    
    def get_reaction_by_reaction(self, reaction: dict) -> Optional[dict]:
        """
        Look up a reaction by substructure match.
        
        Args:
            reaction: dict with 'reactants' and 'products' keys (list of species dicts)
                       each species dict has 'adjacency_list' key
            
        Returns:
            Matched reaction dict or None if not found
        """
        from rmgpu.molecule.molecule import Molecule
        
        # Get all reactions with their species
        reactions_df = self._load_reactions()
        species_df = self._load_species()
        species_by_reaction: dict[int, list[dict]] = {}
        for _, row in species_df.iterrows():
            reaction_id = int(row["library_reaction_id"])
            species_by_reaction.setdefault(reaction_id, []).append(row.to_dict())
        
        for _, row in reactions_df.iterrows():
            reaction_id = int(row["id"])
            reaction_species = species_by_reaction.get(reaction_id, [])
            
            # Build reaction spec lists from DB
            db_reactants = []
            db_products = []
            for spec in reaction_species:
                if spec["role"] == "reactant":
                    db_reactants.append(spec["species_label"])
                else:
                    db_products.append(spec["species_label"])
            
            # Build query spec lists
            q_reactants = []
            for sp in reaction.get("reactants", []):
                if "adjacency_list" in sp:
                    q_reactants.append(sp["adjacency_list"])
            q_products = []
            for sp in reaction.get("products", []):
                if "adjacency_list" in sp:
                    q_products.append(sp["adjacency_list"])
            
            # Compare spec lists (exact match on species labels)
            if sorted(db_reactants) == sorted(q_reactants) and sorted(db_products) == sorted(q_products):
                return {
                    "id": reaction_id,
                    "label": row["label"],
                    "reactants": db_reactants,
                    "products": db_products,
                }
        
        return None

    def is_in_library(self, label: str) -> bool:
        """Return True if a reaction label exists in any kinetics library."""
        df = self._load_reactions()
        return len(df[df["label"] == label]) > 0
    
    def get_library_names(self) -> list[str]:
        """Return all kinetics library names."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM kinetics_libraries_table")
            return [row[0] for row in cur.fetchall()]
    
    def get_library_reaction_count(self, library_name: str) -> int:
        """Return the number of reactions in a specific library."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT COUNT(*)
                FROM kinetics_library_reactions_table
                WHERE library_id = (SELECT id FROM kinetics_libraries_table WHERE name = ?)
            """, (library_name,))
            return cur.fetchone()[0]
    
    def get_family_names(self) -> list[str]:
        """Return all family names."""
        df = self._load_families()
        return df["name"].tolist()
    
    def get_family_definition(self, name: str) -> Optional[dict]:
        """Return a family definition with parsed template and recipe."""
        family = self.get_family_by_name(name)
        if family is None:
            return None
        
        import ast
        result = {
            "name": family["name"],
            "short_description": family.get("short_description", ""),
            "long_description": family.get("long_description", ""),
            "reversible": bool(family.get("reversible", False)),
            "reverse_map": ast.literal_eval(family["reverse_map"]) if family.get("reverse_map") else {},
            "reactant_num": family.get("reactant_num"),
            "product_num": family.get("product_num"),
            "auto_generated": bool(family.get("auto_generated", False)),
        }
        
        # Parse template and recipe from stored string representations
        if family.get("template"):
            result["template"] = ast.literal_eval(family["template"])
        if family.get("recipe"):
            result["recipe"] = ast.literal_eval(family["recipe"])
        
        return result
    
    def get_family_groups(self, family_name: str) -> list[dict]:
        """Return all groups for a family."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT g.label, g.group_adj_list, g.short_description, g.long_description
                FROM kinetics_family_groups_table g
                JOIN kinetics_families_table f ON g.family_id = f.id
                WHERE f.name = ?
            """, (family_name,))
            return [
                {
                    "label": row[0],
                    "group_adj_list": row[1],
                    "short_description": row[2],
                    "long_description": row[3],
                }
                for row in cur.fetchall()
            ]

    def get_rate_model(self, reaction_id: int):
        """Build the rmgpu rate model (SI units) for a library reaction.

        Delegates to rmgpu.data.kinetics.assemble_rate_model, which reads the
        rate tables for ``reaction_id`` and returns an Arrhenius /
        MultiArrhenius / Lindemann / Troe / PDepArrhenius model, or None when
        the reaction carries no stored rate (duplicate/seed) or only Chebyshev
        (coefficients not stored by rmgdb - documented gap).
        """
        from rmgpu.data.kinetics import assemble_rate_model
        return assemble_rate_model(str(self.db_path), int(reaction_id))

    # ------------------------------------------------------------------
    # Raw-row readers for the job-02 gate (content hash + rate round-trip).
    # These expose the flat rmgdb tables in the shape the shared normalizer
    # (gates/normalizer.py) consumes, so the gate compares rmgdb data to
    # RMG-Py on identical canonical structures.
    # ------------------------------------------------------------------

    def get_library_reaction_ids(self, library_name: str) -> list[int]:
        """Reaction ids of a library, in stored (file) order.

        rmgdb allocates ``kinetics_library_reactions_table.id`` by exec-order
        of each library's reactions.py, and RMG-Py assigns ``index`` in the
        same per-library file order - so this list lines up positionally with
        RMG-Py's ordered ``entries.values()`` for the same library.
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """SELECT r.id FROM kinetics_library_reactions_table r
                   JOIN kinetics_libraries_table l ON l.id = r.library_id
                   WHERE l.name = ? ORDER BY r.id""",
                (library_name,),
            )
            return [row[0] for row in cur.fetchall()]

    @staticmethod
    def _effs(cur, reaction_id: int) -> dict:
        cur.execute(
            "SELECT species_label, efficiency FROM kinetics_efficiencies_table"
            " WHERE library_reaction_id = ?",
            (reaction_id,),
        )
        return {s: v for s, v in cur.fetchall() if v is not None}

    @staticmethod
    def _one(cur, table: str, reaction_id: int) -> Optional[dict]:
        cur.execute(
            f"SELECT * FROM {table} WHERE library_reaction_id = ? LIMIT 1",
            (reaction_id,),
        )
        cols = [d[0] for d in cur.description]
        r = cur.fetchone()
        return dict(zip(cols, r)) if r else None

    @staticmethod
    def _many(cur, table: str, reaction_id: int) -> list[dict]:
        cur.execute(f"SELECT * FROM {table} WHERE library_reaction_id = ?", (reaction_id,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def get_raw_reaction(self, reaction_id: int) -> Optional[dict]:
        """Raw reaction row for ``reaction_id`` in the normalizer's shape.

        Carries ``label`` and exactly one of the model payloads: ``arr``
        (list), ``thirdbody``, ``lindemann``, ``troe``, ``chebyshev``, or
        ``pdep`` (list of header rows, each with its ``pressures``). Matches
        what ``gates/normalizer.normalize_kinetics_entry`` and
        ``_raw_from_rmgpy_kinetics`` both produce, so the two sides normalize
        identically.
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT label FROM kinetics_library_reactions_table WHERE id = ?",
                (reaction_id,),
            )
            label = cur.fetchone()
            if label is None:
                return None
            row: dict = {"label": label[0]}
            troe = self._one(cur, "kinetics_troe_table", reaction_id)
            lind = self._one(cur, "kinetics_lindemann_table", reaction_id)
            tb = self._one(cur, "kinetics_third_body_table", reaction_id)
            cheb = self._one(cur, "kinetics_chebyshev_table", reaction_id)
            pdeps = self._many(cur, "kinetics_pdep_arrhenius_table", reaction_id)
            arrs = self._many(cur, "kinetics_arrhenius_table", reaction_id)
            if troe is not None:
                row["troe"] = troe
                row["efficiencies"] = self._effs(cur, reaction_id)
            elif lind is not None:
                row["lindemann"] = lind
                row["efficiencies"] = self._effs(cur, reaction_id)
            elif tb is not None:
                row["thirdbody"] = tb
                row["efficiencies"] = self._effs(cur, reaction_id)
            elif cheb is not None:
                row["chebyshev"] = cheb
            elif pdeps:
                hdrs = []
                for p in pdeps:
                    cur.execute(
                        "SELECT * FROM kinetics_pdep_arrhenius_pressures_table"
                        " WHERE pdep_id = ?",
                        (p["id"],),
                    )
                    cols = [d[0] for d in cur.description]
                    pts = [dict(zip(cols, r)) for r in cur.fetchall()]
                    hdrs.append(
                        {"Tmin_val": p["Tmin_val"], "Tmax_val": p["Tmax_val"],
                         "pressures": pts}
                    )
                row["pdep"] = hdrs
            elif arrs:
                row["arr"] = arrs
            return row

    def get_raw_library_reactions(self, library_name: str) -> list[Optional[dict]]:
        """Raw rows (normalizer shape) for every reaction in a library,
        in stored order - lines up positionally with RMG-Py's entries."""
        return [self.get_raw_reaction(rid) for rid in self.get_library_reaction_ids(library_name)]


class TransportDB:
    """Facade for transport property data (Lennard-Jones parameters + collision data).
    
    Consumed by jobs 06 (core/edge loop) and 07 (master equation) for collision
    parameter estimates.
    """

    def __init__(self, db_path: str | Path = "/home/jackson/rmgpu/rmgdb/db/transport.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self._df: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        """Load transport_libraries_view into a DataFrame (cached)."""
        if self._df is None:
            self._df = pd.read_sql(
                "SELECT * FROM transport_libraries_view",
                f"sqlite:///{self.db_path}",
            )
        return self._df

    def get_entry_count(self) -> int:
        """Return the total number of transport entries."""
        return len(self._load())

    def get_entry_by_label(self, label: str) -> Optional[TransportEntry]:
        """Look up a transport entry by label."""
        df = self._load()
        matches = df[df["label"] == label]
        if matches.empty:
            return None
        return self._row_to_entry(matches.iloc[0])

    def get_entries_by_library(self, library_name: str) -> list[TransportEntry]:
        """Return all entries from a named library."""
        df = self._load()
        matches = df[df["name"] == library_name]
        return [self._row_to_entry(row) for _, row in matches.iterrows()]

    def get_entry_by_adjlist(self, adjlist: str) -> Optional[TransportEntry]:
        """Look up a transport entry by adjacency list (exact match)."""
        df = self._load()
        matches = df[df["adjacency_list"] == adjlist]
        if matches.empty:
            return None
        return self._row_to_entry(matches.iloc[0])

    def _row_to_entry(self, row: pd.Series) -> TransportEntry:
        """Convert a pandas Series to a TransportEntry."""
        def _float_or_none(key: str) -> Optional[float]:
            val = row.get(key)
            if pd.isna(val):
                return None
            return float(val)

        return TransportEntry(
            label=str(row["label"]),
            adjacency_list=str(row.get("adjacency_list", "")),
            library_name=str(row.get("name", "")),
            shape_index=float(row.get("shapeIndex", 0) or 0),
            epsilon=float(row.get("epsilon", 0) or 0),
            epsilon_unit=str(row.get("epsilon_unit", "K") or "K"),
            sigma=float(row.get("sigma", 0) or 0),
            sigma_unit=str(row.get("sigma_unit", "angstroms") or "angstroms"),
            dipole_moment=float(row.get("dipoleMoment", 0) or 0),
            dipole_moment_unit=str(row.get("dipoleMoment_unit", "De") or "De"),
            polarizability=float(row.get("polarizability", 0) or 0),
            polarizability_unit=str(row.get("polarizability_unit", "angstroms^3") or "angstroms^3"),
            rot_relax_coll_num=float(row.get("rotrelaxcollnum", 1.0) or 1.0),
        )


class StatMechDB:
    """Facade for statistical mechanics data (group frequencies + libraries).
    
    Consumed by job 07 (statmech + master equation) for group frequency
    contributions and conformer data.
    """

    def __init__(self, db_path: str | Path = "/home/jackson/rmgpu/rmgdb/db/statmech.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self._df_libraries: Optional[pd.DataFrame] = None
        self._df_groups: Optional[pd.DataFrame] = None

    def _load_libraries(self) -> pd.DataFrame:
        """Load statmech_libraries_view into a DataFrame (cached)."""
        if self._df_libraries is None:
            self._df_libraries = pd.read_sql(
                "SELECT * FROM statmech_libraries_view",
                f"sqlite:///{self.db_path}",
            )
        return self._df_libraries

    def _load_groups(self) -> pd.DataFrame:
        """Load statmech_groups_view into a DataFrame (cached)."""
        if self._df_groups is None:
            self._df_groups = pd.read_sql(
                "SELECT * FROM statmech_groups_view",
                f"sqlite:///{self.db_path}",
            )
        return self._df_groups

    def get_library_count(self) -> int:
        """Return the number of statmech library entries."""
        return len(self._load_libraries())

    def get_group_count(self) -> int:
        """Return the number of statmech group entries."""
        return len(self._load_groups())

    def get_entry_by_label(self, label: str) -> Optional[StatMechEntry]:
        """Look up a statmech entry by label."""
        df = self._load_libraries()
        matches = df[df["label"] == label]
        if matches.empty:
            return None
        return self._row_to_entry(matches.iloc[0])

    @staticmethod
    def _grp(row: pd.Series) -> StatMechGroup:
        def _fl(key: str) -> float:
            val = row.get(key)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return 0.0
            return float(val)

        def _fi(key: str) -> int:
            val = row.get(key)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return 1
            return int(val)

        return StatMechGroup(
            name=str(row.get("name", "")),
            label=str(row["label"]),
            group=str(row.get("group", "")),
            symmetry=_fi("symmetry"),
            lower=_fl("lower"),
            upper=_fl("upper"),
            degeneracy=_fi("degeneracy"),
        )

    def get_groups(self) -> list[StatMechGroup]:
        """Return all group (characteristic-frequency) entries."""
        df = self._load_groups()
        return [self._grp(r) for _, r in df.iterrows()]

    def get_group_by_label(self, label: str) -> Optional[StatMechGroup]:
        """Look up a group entry by label."""
        df = self._load_groups()
        matches = df[df["label"] == label]
        if matches.empty:
            return None
        return self._grp(matches.iloc[0])

    def _row_to_entry(self, row: pd.Series) -> StatMechEntry:
        """Convert a pandas Series to a StatMechEntry."""

        def _fl(key: str) -> Optional[float]:
            val = row.get(key)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            return float(val)

        def _fi(key: str) -> Optional[int]:
            val = row.get(key)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            return int(val)

        freqs = []
        funit = row.get("harmonic_freq_unit")
        for i in range(1, 13):
            val = row.get(f"harmonic_freq_{i}")
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                freqs.append(float(val))

        inertia = None
        ix, iy, iz = _fl("inertia_x"), _fl("inertia_y"), _fl("inertia_z")
        if ix is not None and iy is not None and iz is not None:
            inertia = [ix, iy, iz]

        return StatMechEntry(
            label=str(row["label"]),
            adjacency_list=str(row.get("adjacency_list", "")),
            library_name=str(row.get("name", "")),
            energy=float(row.get("energy", 0) or 0),
            energy_unit=str(row.get("energy_unit", "kcal/mol") or "kcal/mol"),
            spin_multiplicity=_fi("spin_multiplicity"),
            optical_isomers=_fi("optical_isomers"),
            mass=_fl("mass"),
            mass_unit=str(row.get("mass_unit") or "") if row.get("mass_unit") else None,
            nonlinear_inertia=inertia,
            nonlinear_inertia_unit=str(row.get("nonlinear_inertia_unit") or "") if row.get("nonlinear_inertia_unit") else None,
            nonlinear_symmetry=_fi("nonlinear_symmetry"),
            linear_inertia=_fl("linear_inertia"),
            linear_inertia_unit=str(row.get("linear_inertia_unit") or "") if row.get("linear_inertia_unit") else None,
            linear_symmetry=_fi("linear_symmetry"),
            harmonic_freq=freqs,
            harmonic_freq_unit=str(funit) if funit else None,
        )


class SolvationDB:
    """Facade for solvation data (groups + libraries).
    
    Consumed by job 11 (solvation plugin + liquid reactors) for solvation
    group contributions and solute/solvent data.
    """

    def __init__(self, db_path: str | Path = "/home/jackson/rmgpu/rmgdb/db/solvation.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self._df_groups: Optional[pd.DataFrame] = None
        self._df_solutes: Optional[pd.DataFrame] = None
        self._df_solvents: Optional[pd.DataFrame] = None

    def _load_groups(self) -> pd.DataFrame:
        """Load solvation groups into a DataFrame (cached)."""
        if self._df_groups is None:
            self._df_groups = pd.read_sql(
                "SELECT * FROM groups_table",
                f"sqlite:///{self.db_path}",
            )
        return self._df_groups

    def _load_solutes(self) -> pd.DataFrame:
        """Load solute libraries into a DataFrame (cached).

        Uses the solute_libraries_view (solute_libraries_table + solute_data_table)
        which carries the Abraham S/B/E/L/A/V parameters.
        """
        if self._df_solutes is None:
            self._df_solutes = pd.read_sql(
                "SELECT * FROM solute_libraries_view",
                f"sqlite:///{self.db_path}",
            )
        return self._df_solutes

    def _load_solvents(self) -> pd.DataFrame:
        """Load solvent libraries into a DataFrame (cached)."""
        if self._df_solvents is None:
            self._df_solvents = pd.read_sql(
                "SELECT * FROM solvent_libraries_table",
                f"sqlite:///{self.db_path}",
            )
        return self._df_solvents

    def get_group_count(self) -> int:
        """Return the number of solvation group entries."""
        return len(self._load_groups())

    def get_solute_count(self) -> int:
        """Return the number of solute library entries."""
        return len(self._load_solutes())

    def get_solvent_count(self) -> int:
        """Return the number of solvent library entries."""
        return len(self._load_solvents())

    def get_group_by_label(self, label: str) -> Optional[SolvationEntry]:
        """Look up a solvation group by label."""
        df = self._load_groups()
        matches = df[df["label"] == label]
        if matches.empty:
            return None
        row = matches.iloc[0]
        return SolvationEntry(
            label=str(row["label"]),
            group=str(row.get("group", "")),
            short_description=str(row.get("short_description", "")),
            long_description=str(row.get("long_description", "")),
            solute_pointer=row.get("solute_pointer"),
        )

    def get_solute_by_label(self, label: str) -> Optional[SoluteLibraryEntry]:
        """Look up a solute library entry by label (Abraham params from the view)."""
        df = self._load_solutes()
        matches = df[df["label"] == label]
        if matches.empty:
            return None
        row = matches.iloc[0]

        def _fl(key: str) -> float:
            val = row.get(key)
            return 0.0 if (val is None or (isinstance(val, float) and pd.isna(val))) else float(val)

        return SoluteLibraryEntry(
            label=str(row["label"]),
            name=str(row.get("name", "")),
            molecule=str(row.get("molecule", "")),
            short_description=str(row.get("short_description", "")),
            long_description=str(row.get("long_description", "")),
            S=_fl("S"), B=_fl("B"), E=_fl("E"), L=_fl("L"), A=_fl("A"), V=_fl("V"),
        )

    def get_all_groups(self) -> list[SolvationEntry]:
        """Return all solvation groups."""
        df = self._load_groups()
        return [
            SolvationEntry(
                label=str(row["label"]),
                group=str(row.get("group", "")),
                short_description=str(row.get("short_description", "")),
                long_description=str(row.get("long_description", "")),
                solute_pointer=row.get("solute_pointer"),
            )
            for _, row in df.iterrows()
        ]

    def get_solvent_by_label(self, label: str) -> Optional[SolventLibraryEntry]:
        """Look up a solvent library entry by label (Abraham + viscosity params)."""
        df = self._load_solvents()
        matches = df[df["label"] == label]
        if matches.empty:
            return None
        row = matches.iloc[0]

        def _fn(key: str) -> Optional[float]:
            val = row.get(key)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            return float(val)

        return SolventLibraryEntry(
            label=str(row["label"]),
            name=str(row.get("name", "")),
            molecule=str(row.get("molecule", "")),
            s_g=_fn("s_g"), b_g=_fn("b_g"), e_g=_fn("e_g"), l_g=_fn("l_g"),
            a_g=_fn("a_g"), c_g=_fn("c_g"),
            s_h=_fn("s_h"), b_h=_fn("b_h"), e_h=_fn("e_h"), l_h=_fn("l_h"),
            a_h=_fn("a_h"), c_h=_fn("c_h"),
            A=_fn("A"), B=_fn("B"), C=_fn("C"), D=_fn("D"), E=_fn("E"),
            alpha=_fn("alpha"), beta=_fn("beta"), eps=_fn("eps"), n=_fn("n"),
            name_in_coolprop=str(row.get("name_in_coolprop")) if row.get("name_in_coolprop") else None,
            dGsolvCount=_fn("dGsolvCount"),
            dGsolvMAE_val=_fn("dGsolvMAE_val"),
            dGsolvMAE_unit=str(row.get("dGsolvMAE_unit")) if row.get("dGsolvMAE_unit") else None,
            dHsolvCount=_fn("dHsolvCount"),
            dHsolvMAE_val=_fn("dHsolvMAE_val"),
            dHsolvMAE_unit=str(row.get("dHsolvMAE_unit")) if row.get("dHsolvMAE_unit") else None,
        )

    def get_all_solvents(self) -> list[SolventLibraryEntry]:
        """Return all solvent library entries."""
        df = self._load_solvents()
        out = []
        for _, row in df.iterrows():
            e = self.get_solvent_by_label(str(row["label"]))
            if e is not None:
                out.append(e)
        return out
