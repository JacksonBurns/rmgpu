"""ThermoDB and KineticsDB facades over rmgdb SQLite databases."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from rmgpu.data.entries import ThermoEntry


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

    def get_entry_by_label(self, label: str) -> Optional[ThermoEntry]:
        """Look up a thermo entry by its label."""
        df = self._load()
        matches = df[df["label"] == label]
        if matches.empty:
            return None
        return self._row_to_entry(matches.iloc[0])

    def get_entries_by_library(self, library_name: str) -> list[ThermoEntry]:
        """Return all entries from a named library."""
        df = self._load()
        matches = df[df["name"] == library_name]
        return [self._row_to_entry(row) for _, row in matches.iterrows()]

    def is_in_library(self, label: str, library_name: str) -> bool:
        """Return True if a label exists in the given library."""
        df = self._load()
        return len(df[(df["label"] == label) & (df["name"] == library_name)]) > 0

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
            Cp0=_float_or_none("Cp0"),
            CpInf=_float_or_none("CpInf"),
            a0=_float_or_none("a0"),
            a1=_float_or_none("a1"),
            a2=_float_or_none("a2"),
            a3=_float_or_none("a3"),
            B=_float_or_none("B"),
            H0=_float_or_none("H0"),
            S0=_float_or_none("S0"),
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
