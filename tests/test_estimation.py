"""Tests for the estimation resolvers.

Tests cover:
- Library hit wins over ML
- ML hit when no library
- MLCoverageError when neither (and error is NOT swallowed)
- The counts object reflects the split exactly
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from rmgpu.data.estimation import EstimationCounts, estimate_thermo, estimate_kinetics
from rmgpu.ml.thermo_estimator import ThermoPrediction, CpModel


# --- Mock ML and DB ---

class MockThermoML:
    def __init__(self):
        self.called = False

    def covers(self, smiles):
        return True

    def predict(self, smiles):
        self.called = True
        return ThermoPrediction(
            Hf298=100.0,
            S298=50.0,
            Cp_model=CpModel(T=np.array([]), Cp=np.array([]), wilhoit=None),
            uncertainties={"source": "mock"},
        )


class MockMLNoCoverage:
    called = False

    def covers(self, smiles):
        return False

    def predict(self, smiles):
        raise Exception("should not be called")


@dataclass
class MockML:
    thermo: MockThermoML | MockMLNoCoverage = None

    def __post_init__(self):
        if self.thermo is None:
            self.thermo = MockThermoML()


@dataclass
class MockThermoDB:
    entries: dict = field(default_factory=dict)

    def get_entry_grouped_by_label(self, label):
        return self.entries.get(label)


class MockThermoEntry:
    def __init__(self, H298=50.0, S298=25.0, Tdata=None, Cpdata=None, label="test"):
        self.H298 = H298
        self.S298 = S298
        self.Tdata = Tdata or []
        self.Cpdata = Cpdata or []
        self.label = label


@dataclass
class MockKineticsDB:
    reactions: dict = field(default_factory=dict)

    def get_reaction_by_reaction(self, reaction):
        return self.reactions.get(reaction.get("label"))

    def get_rate_model(self, reaction_id):
        return MockRateModel()


class MockRateModel:
    pass


class MockKineticsML:
    def covers(self, smiles):
        return True

    def predict(self, smiles):
        return MockKineticsPrediction()


class MockKineticsPrediction:
    A = 1.0
    n = 0.0
    Ea_J_mol = 1000.0
    Tmin = 300.0
    Tmax = 1000.0
    degeneracy = 2.0
    source = "mock"


@dataclass
class MockMLKinetics:
    kinetics: MockKineticsML = field(default_factory=MockKineticsML)


# --- Tests for estimate_thermo ---

def test_estimate_thermo_library_hit():
    """Library hit wins over ML."""
    db = MockThermoDB(entries={"test": MockThermoEntry(H298=50.0, S298=25.0)})
    ml = MockML()
    counts = EstimationCounts()

    species = {"label": "test", "smiles": "CC"}
    result = estimate_thermo(species, db, ml, counts)

    assert result.Hf298 == 50.0
    assert result.S298 == 25.0
    assert counts.library_hits == 1
    assert counts.ml_hits == 0
    assert counts.coverage_errors == 0
    assert not ml.thermo.called


def test_estimate_thermo_ml_hit_when_no_library():
    """ML hit when no library entry."""
    db = MockThermoDB(entries={})
    ml = MockML()
    counts = EstimationCounts()

    species = {"label": "unknown", "smiles": "CC"}
    result = estimate_thermo(species, db, ml, counts)

    assert result.Hf298 == 100.0
    assert result.S298 == 50.0
    assert counts.library_hits == 0
    assert counts.ml_hits == 1
    assert counts.coverage_errors == 0
    assert ml.thermo.called


def test_estimate_thermo_ml_coverage_error():
    """MLCoverageError when ML doesn't cover."""
    db = MockThermoDB(entries={})
    ml = MockML()
    ml.thermo = MockMLNoCoverage()
    counts = EstimationCounts()

    species = {"label": "unknown", "smiles": "CC"}

    with pytest.raises(Exception) as exc_info:
        estimate_thermo(species, db, ml, counts)

    assert "does not cover" in str(exc_info.value)
    assert counts.library_hits == 0
    assert counts.ml_hits == 0
    assert counts.coverage_errors == 1


def test_estimate_thermo_counts_object_reflects_split():
    """The counts object reflects the split exactly."""
    db = MockThermoDB(entries={"lib": MockThermoEntry()})
    ml = MockML()
    counts = EstimationCounts()

    # Library hit
    estimate_thermo({"label": "lib", "smiles": "CC"}, db, ml, counts)

    # ML hit
    estimate_thermo({"label": "ml", "smiles": "CC"}, db, ml, counts)

    # Coverage error
    db2 = MockThermoDB(entries={})
    ml2 = MockML()
    ml2.thermo = MockMLNoCoverage()
    with pytest.raises(Exception):
        estimate_thermo({"label": "err", "smiles": "CC"}, db2, ml2, counts)

    assert counts.library_hits == 1
    assert counts.ml_hits == 1
    assert counts.coverage_errors == 1
    total = counts.library_hits + counts.ml_hits + counts.coverage_errors
    assert total == 3  # 3 calls, 3 outcomes


# --- Tests for estimate_kinetics ---

def test_estimate_kinetics_library_hit():
    """Library hit wins over ML."""
    reaction = {"label": "rxn1", "reactants": [], "products": []}
    db = MockKineticsDB(reactions={"rxn1": {"id": 1}})
    ml = MockMLKinetics()
    counts = EstimationCounts()

    result, deg = estimate_kinetics(reaction, db, ml, counts)

    assert result is not None
    assert deg == 1.0
    assert counts.library_hits == 1
    assert counts.ml_hits == 0


def test_estimate_kinetics_ml_hit_when_no_library():
    """ML hit when no library entry."""
    reaction = {"label": "unknown", "reactants": [{"smiles": "CC"}], "products": [{"smiles": "C"}]}
    db = MockKineticsDB(reactions={})
    ml = MockMLKinetics()
    counts = EstimationCounts()

    result, deg = estimate_kinetics(reaction, db, ml, counts)

    assert result is not None
    assert deg == 2.0
    assert counts.library_hits == 0
    assert counts.ml_hits == 1
    assert counts.coverage_errors == 0
