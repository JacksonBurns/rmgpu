"""Tests for database facades."""

import pytest
import tempfile
import os
from pathlib import Path

from rmgpu.db import Databases
from rmgpu.db.loaders import ThermoDB, KineticsDB, TransportDB, StatMechDB, SolvationDB


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    return Path(tempfile.mkdtemp()) / "test.db"


@pytest.fixture
def create_temp_db(temp_db_path):
    """Create a temporary database file."""
    import sqlite3
    conn = sqlite3.connect(str(temp_db_path))
    cursor = conn.cursor()
    # Create minimal schema
    cursor.execute("""
        CREATE TABLE thermo_libraries_view (
            id INTEGER PRIMARY KEY,
            name TEXT,
            label TEXT,
            H298 REAL,
            S298 REAL,
            short_description TEXT,
            long_description TEXT,
            Tdata_unit TEXT,
            Cpdata_unit TEXT,
            H298_unit TEXT,
            S298_unit TEXT
        )
    """)
    conn.commit()
    conn.close()
    return temp_db_path


class TestDatabasesAggregate:
    """Test the Databases aggregate facade."""
    
    def test_from_config_creates_all_facades(self):
        """Test that from_config creates all facades."""
        # Use the default paths that exist
        databases = Databases.from_config({})
        
        assert databases.thermo is not None
        assert databases.kinetics is not None
        assert databases.transport is not None
        assert databases.statmech is not None
        assert databases.solvation is not None
    
    def test_from_config_uses_defaults(self):
        """Test that from_config uses default paths when config is empty."""
        databases = Databases.from_config({})
        
        assert databases.thermo is not None
        assert databases.kinetics is not None
        assert databases.transport is not None
        assert databases.statmech is not None
        assert databases.solvation is not None


class TestTransportDB:
    """Test the TransportDB facade."""
    
    @pytest.fixture(autouse=True)
    def setup_db(self):
        """Set up database."""
        databases = Databases.from_config({})
        self.db = databases.transport
    
    def test_get_entry_count(self):
        """Test getting entry count."""
        count = self.db.get_entry_count()
        assert isinstance(count, int)
    
    def test_get_entry_by_label(self):
        """Test getting entry by label."""
        entry = self.db.get_entry_by_label("test")
        # Entry may or may not exist
        assert entry is None or hasattr(entry, 'label')
    
    def test_get_entries_by_library(self):
        """Test getting entries by library."""
        entries = self.db.get_entries_by_library("test")
        assert isinstance(entries, list)
    
    def test_get_entry_by_adjlist(self):
        """Test getting entry by adjacency list."""
        entry = self.db.get_entry_by_adjlist("test")
        # Entry may or may not exist
        assert entry is None or hasattr(entry, 'label')


class TestStatMechDB:
    """Test the StatMechDB facade."""
    
    @pytest.fixture(autouse=True)
    def setup_db(self):
        """Set up database."""
        databases = Databases.from_config({})
        self.db = databases.statmech
    
    def test_get_library_count(self):
        """Test getting library count."""
        count = self.db.get_library_count()
        assert isinstance(count, int)
    
    def test_get_group_count(self):
        """Test getting group count."""
        count = self.db.get_group_count()
        assert isinstance(count, int)
    
    def test_get_entry_by_label(self):
        """Test getting entry by label."""
        entry = self.db.get_entry_by_label("test")
        # Entry may or may not exist
        assert entry is None or hasattr(entry, 'label')
    
    def test_get_groups(self):
        """Test getting all groups."""
        groups = self.db.get_groups()
        assert isinstance(groups, list)


class TestSolvationDB:
    """Test the SolvationDB facade."""
    
    @pytest.fixture(autouse=True)
    def setup_db(self):
        """Set up database."""
        databases = Databases.from_config({})
        self.db = databases.solvation
    
    def test_get_group_count(self):
        """Test getting group count."""
        count = self.db.get_group_count()
        assert isinstance(count, int)
    
    def test_get_solute_count(self):
        """Test getting solute count."""
        count = self.db.get_solute_count()
        assert isinstance(count, int)
    
    def test_get_solvent_count(self):
        """Test getting solvent count."""
        count = self.db.get_solvent_count()
        assert isinstance(count, int)
    
    def test_get_group_by_label(self):
        """Test getting group by label."""
        entry = self.db.get_group_by_label("test")
        # Entry may or may not exist
        assert entry is None or hasattr(entry, 'label')
    
    def test_get_solute_by_label(self):
        """Test getting solute by label."""
        entry = self.db.get_solute_by_label("test")
        # Entry may or may not exist
        assert entry is None or hasattr(entry, 'label')
    
    def test_get_all_groups(self):
        """Test getting all groups."""
        groups = self.db.get_all_groups()
        assert isinstance(groups, list)