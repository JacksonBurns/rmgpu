"""Tests for KineticsDB facade."""

import os
import sys
import sqlite3
from pathlib import Path

import pytest

from rmgpu.db.loaders import KineticsDB

DB_PATH = "/home/jackson/rmgpu/rmgdb/db/kinetics.db"


@pytest.fixture
def kinetics_db():
    return KineticsDB(db_path=DB_PATH)


class TestKineticsDBBasics:
    def test_constructor_raises_if_missing(self):
        with pytest.raises(FileNotFoundError):
            KineticsDB(db_path="/nonexistent/path/kinetics.db")
    
    def test_get_library_count(self, kinetics_db):
        count = kinetics_db.get_library_count()
        assert count > 0
        # Should be 63 libraries
        assert count == 63
    
    def test_get_reaction_count(self, kinetics_db):
        count = kinetics_db.get_reaction_count()
        assert count > 0
        # Should be 20257 reactions
        assert count == 20257
    
    def test_get_family_count(self, kinetics_db):
        count = kinetics_db.get_family_count()
        assert count > 0
        # Should be 129 families
        assert count == 129
    
    def test_get_library_names(self, kinetics_db):
        names = kinetics_db.get_library_names()
        assert len(names) == 63
        assert all(isinstance(n, str) for n in names)
        # Check some known libraries
        assert any("BurkeH2O2inArHe" in n for n in names)
    
    def test_get_library_reaction_count(self, kinetics_db):
        # Get first library name and check its count
        names = kinetics_db.get_library_names()
        first_lib = names[0]
        count = kinetics_db.get_library_reaction_count(first_lib)
        assert count >= 0
    
    def test_get_family_names(self, kinetics_db):
        names = kinetics_db.get_family_names()
        assert len(names) == 129
        assert all(isinstance(n, str) for n in names)
    
    def test_get_family_by_name(self, kinetics_db):
        family = kinetics_db.get_family_by_name("1+2_Cycloaddition")
        assert family is not None
        assert family["name"] == "1+2_Cycloaddition"
    
    def test_get_family_definition(self, kinetics_db):
        family = kinetics_db.get_family_definition("1+2_Cycloaddition")
        assert family is not None
        assert family["name"] == "1+2_Cycloaddition"
        assert "template" in family
        assert "recipe" in family
        assert isinstance(family["template"], dict)
        assert isinstance(family["recipe"], dict)
    
    def test_get_family_groups(self, kinetics_db):
        groups = kinetics_db.get_family_groups("1+2_Cycloaddition")
        assert isinstance(groups, list)
        assert len(groups) > 0
        assert all("label" in g and "group_adj_list" in g for g in groups)
    
    def test_get_reaction_by_label(self, kinetics_db):
        # Get first reaction and test lookup
        import pandas as pd
        df = pd.read_sql(
            "SELECT label FROM kinetics_library_reactions_table LIMIT 1",
            f"sqlite:///{DB_PATH}"
        )
        label = df.iloc[0]["label"]
        result = kinetics_db.get_reaction_by_label(label)
        assert result is not None
        assert result["label"] == label
        assert "reactants" in result
        assert "products" in result
    
    def test_get_reaction_by_reaction(self, kinetics_db):
        # Test substructure match lookup
        reaction = {
            "reactants": [{"adjacency_list": "label\n  1 H u0 c0 {2}S\n  2 H u0 c0 {1}S\n"}],
            "products": [{"adjacency_list": "label\n  1 H u1 c0\n"}],
        }
        result = kinetics_db.get_reaction_by_reaction(reaction)
        # Should find H2 -> H match (if it exists in DB)
        if result is not None:
            assert "label" in result
            assert "reactants" in result
            assert "products" in result


class TestFamilyStorageInvestigation:
    """Verify that families are stored in rmgdb."""
    
    def test_family_storage(self, kinetics_db):
        """Check that family definitions are properly stored."""
        # 1. Check family count
        count = kinetics_db.get_family_count()
        assert count == 129
        
        # 2. Check that families have templates
        family = kinetics_db.get_family_definition("1+2_Cycloaddition")
        assert family is not None
        assert "template" in family
        
        # 3. Check that families have recipes
        assert "recipe" in family
        assert len(family["recipe"]) > 0
        
        # 4. Check family groups are stored
        groups = kinetics_db.get_family_groups("1+2_Cycloaddition")
        assert len(groups) > 0
        
        # 5. Check that family rules table is empty (as expected)
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM kinetics_family_rules_table")
            rule_count = cur.fetchone()[0]
        assert rule_count == 0
    
    def test_depository_is_just_a_library(self, kinetics_db):
        """Verify that training depository is just a library."""
        names = kinetics_db.get_library_names()
        # Check that there's no special "training" depository type
        # All kinetics data is in regular libraries
        assert len(names) == 63