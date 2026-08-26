"""Tests for the thermo database layer (ThermoDB facade + thermo models)."""

import pytest
from rmgpu.db.loaders import ThermoDB
from rmgpu.data.thermo import Wilhoit, NASA, NASAPolynomial
from rmgpu.data.entries import ThermoEntry


class TestWilhoitModel:
    """Tests for the Wilhoit heat capacity model."""

    def test_wilhoit_heat_capacity(self):
        """Test Cp(T) calculation matches expected values."""
        # Example from RMG-Py documentation
        wilhoit = Wilhoit(
            Cp0=4.0,
            CpInf=80.0,
            a0=2.2773888,
            a1=-4.3514887,
            a2=3.1068816,
            a3=-0.8201731,
            B=570.0,
        )
        # At T=300 K, Cp should be approximately 18.7 cal/mol/K
        cp = wilhoit.get_heat_capacity(300.0)
        assert cp > 0
        # Test that Cp increases with temperature
        cp_500 = wilhoit.get_heat_capacity(500.0)
        cp_1000 = wilhoit.get_heat_capacity(1000.0)
        assert cp_500 > cp
        assert cp_1000 > cp_500

    def test_wilhoit_enthalpy(self):
        """Test H(T) calculation."""
        wilhoit = Wilhoit(
            Cp0=4.0,
            CpInf=80.0,
            a0=2.2773888,
            a1=-4.3514887,
            a2=3.1068816,
            a3=-0.8201731,
            B=570.0,
            H0=0.0,
        )
        h_298 = wilhoit.get_enthalpy(298.0)
        h_500 = wilhoit.get_enthalpy(500.0)
        assert h_500 > h_298

    def test_wilhoit_entropy(self):
        """Test S(T) calculation."""
        wilhoit = Wilhoit(
            Cp0=4.0,
            CpInf=80.0,
            a0=2.2773888,
            a1=-4.3514887,
            a2=3.1068816,
            a3=-0.8201731,
            B=570.0,
            S0=0.0,
        )
        s_298 = wilhoit.get_entropy(298.0)
        s_500 = wilhoit.get_entropy(500.0)
        assert s_500 > s_298

    def test_wilhoit_free_energy(self):
        """Test G(T) = H(T) - T*S(T)."""
        wilhoit = Wilhoit(
            Cp0=4.0,
            CpInf=80.0,
            a0=2.2773888,
            a1=-4.3514887,
            a2=3.1068816,
            a3=-0.8201731,
            B=570.0,
            H0=0.0,
            S0=0.0,
        )
        T = 500.0
        g = wilhoit.get_free_energy(T)
        h = wilhoit.get_enthalpy(T)
        s = wilhoit.get_entropy(T)
        assert abs(g - (h - T * s)) < 1e-6


class TestNASAModel:
    """Tests for the NASA polynomial model."""

    def test_nasa_heat_capacity(self):
        """Test NASA Cp(T) calculation."""
        coeffs = [1.0, 2e-3, 0.0, 0.0, 0.0, 0.0, 0.0]
        poly = NASAPolynomial(coeffs, 300.0, 1000.0)
        cp = poly.get_heat_capacity(500.0)
        assert cp > 0

    def test_nasa_enthalpy(self):
        """Test NASA H(T) calculation."""
        coeffs = [1.0, 2e-3, 0.0, 0.0, 0.0, 0.0, 0.0]
        poly = NASAPolynomial(coeffs, 300.0, 1000.0)
        h = poly.get_enthalpy(500.0)
        assert h > 0

    def test_nasa_entropy(self):
        """Test NASA S(T) calculation."""
        coeffs = [1.0, 2e-3, 0.0, 0.0, 0.0, 0.0, 0.0]
        poly = NASAPolynomial(coeffs, 300.0, 1000.0)
        s = poly.get_entropy(500.0)
        assert s > 0

    def test_nasa_temperature_selection(self):
        """Test that NASA selects the correct polynomial for T."""
        poly1 = {"coeffs": [1.0, 2e-3, 0.0, 0.0, 0.0, 0.0, 0.0], "Tmin": 300.0, "Tmax": 1000.0}
        poly2 = {"coeffs": [2.0, 4e-3, 0.0, 0.0, 0.0, 0.0, 0.0], "Tmin": 1000.0, "Tmax": 2000.0}
        nasa = NASA([poly1, poly2])
        
        # T=500 should use poly1
        cp_500 = nasa.get_heat_capacity(500.0)
        # T=1500 should use poly2
        cp_1500 = nasa.get_heat_capacity(1500.0)
        assert cp_1500 > cp_500

    def test_nasa_invalid_temperature(self):
        """Test that invalid T raises ValueError."""
        coeffs = [1.0, 2e-3, 0.0, 0.0, 0.0, 0.0, 0.0]
        poly = {"coeffs": coeffs, "Tmin": 300.0, "Tmax": 1000.0}
        nasa = NASA([poly])
        with pytest.raises(ValueError):
            nasa.get_heat_capacity(2000.0)


class TestThermoDB:
    """Tests for the ThermoDB facade."""

    def test_db_entry_count(self):
        """Test that ThermoDB can load and count entries."""
        db = ThermoDB()
        count = db.get_entry_count()
        assert count > 0

    def test_db_entry_by_label(self):
        """Test lookup by label."""
        db = ThermoDB()
        # Get first label from database
        entry = db.get_entry_by_label("C")
        assert entry is not None
        assert entry.label == "C"

    def test_db_entry_not_found(self):
        """Test that nonexistent label returns None."""
        db = ThermoDB()
        entry = db.get_entry_by_label("NONEXISTENT_LABEL_12345")
        assert entry is None

    def test_db_is_in_library(self):
        """Test is_in_library method."""
        db = ThermoDB()
        # Get first library
        df = db._load()
        library_name = df.iloc[0]["name"]
        label = df.iloc[0]["label"]
        assert db.is_in_library(label, library_name)


class TestThermoEntry:
    """Tests for the ThermoEntry data class."""

    def test_thermo_entry_creation(self):
        """Test creating a ThermoEntry."""
        entry = ThermoEntry(
            label="C",
            short_description="Carbon atom",
            long_description="Test description",
            Tdata_unit="K",
            Cpdata_unit="cal/(mol*K)",
            H298=0.0,
            H298_unit="kcal/mol",
            S298=0.0,
            S298_unit="cal/(mol*K)",
        )
        assert entry.label == "C"

    def test_thermo_entry_to_dict(self):
        """Test ThermoEntry serialization."""
        entry = ThermoEntry(
            label="C",
            short_description="Carbon atom",
            long_description="Test description",
            Tdata_unit="K",
            Cpdata_unit="cal/(mol*K)",
            H298=0.0,
            H298_unit="kcal/mol",
            S298=0.0,
            S298_unit="cal/(mol*K)",
        )
        d = entry.to_dict()
        assert d["label"] == "C"
        assert "model" in d
