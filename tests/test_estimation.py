"""Tests for the estimation resolvers (job-04 step-05).

Covers the resolver contract:
- library hit wins over ML (ML never called),
- ML hit when no library,
- MLCoverageError when neither (and the error is NOT swallowed),
- the counts object reflects the split exactly (total == attempts),
- library values come out SI (kcal/mol, cal/(mol*K) converted),
- library entry with no usable model is a coverage gap, not a hit,
- kinetics: library hit -> rate model; gap reaction (rate None) -> ML;
  no ML -> MLCoverageError; the degeneracy boundary conversion applies;
- reaction_smiles is authoritative for the ML branch (the '>>' RIGR format).

ML and DB are mocks with exact values; the real-checkpoint behavior is
covered by test_thermo_ml.py / test_kinetics_ml.py / test_ml_base.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from rmgpu.data.estimation import (
    Cal,
    KCAL,
    EstimationCounts,
    estimate_thermo,
    estimate_kinetics,
    reaction_to_smiles,
    species_to_smiles,
)
from rmgpu.data.entries import ThermoEntry
from rmgpu.kinetics.models import Arrhenius
from rmgpu.ml.thermo_estimator import MLCoverageError, ThermoPrediction


# ---------------------------------------------------------------------------
# Mocks (exact values)
# ---------------------------------------------------------------------------


class MockThermoML:
    """A thermo ML that covers exactly the SMILES given to it."""

    def __init__(self, covered=()):
        self.covered = set(covered)
        self.called_with = []

    def covers(self, smiles):
        return smiles in self.covered

    def predict(self, smiles):
        self.called_with.append(smiles)
        return ThermoPrediction(
            Hf298=123.0,
            S298=45.0,
            Cp_model=None,  # the resolver passes the prediction through
            uncertainties={"source": "mock-ml"},
        )


class MockNoThermoML:
    def __init__(self):
        self.called = False

    def covers(self, smiles):
        self.called = True
        return False

    def predict(self, smiles):
        raise AssertionError("predict must not be called when covers() is False")


@dataclass
class MockML:
    thermo: object = field(default_factory=lambda: MockThermoML(covered=("CC",)))
    kinetics: object = field(default_factory=lambda: MockKineticsML())


@dataclass
class MockThermoEntry:
    """Minimal stand-in for rmgpu.data.entries.ThermoEntry (CGS units)."""

    label: str = "lib"
    H298: float | None = 1.0  # kcal/mol
    H298_unit: str = "kcal/mol"
    S298: float | None = 10.0  # cal/(mol*K)
    S298_unit: str = "cal/(mol*K)"
    Tdata: list = field(default_factory=lambda: [300.0, 400.0, 500.0, 600.0,
                                                  800.0, 1000.0, 1500.0])
    Cpdata: list = field(default_factory=lambda: [30.0, 40.0, 50.0, 60.0,
                                                   80.0, 100.0, 130.0])
    Cpdata_unit: str = "cal/(mol*K)"
    Tdata_unit: str = "K"
    nasa_polynomials: list = field(default_factory=list)


@dataclass
class MockThermoDB:
    entries: dict = field(default_factory=dict)

    def get_entry_grouped_by_label(self, label, library=None):
        return self.entries.get(label)


class MockNoModelEntry:
    """A library row with a label but no H298/S298/Cp model at all."""

    label = "empty"
    H298 = None
    H298_unit = "kcal/mol"
    S298 = None
    S298_unit = "cal/(mol*K)"
    Tdata = []
    Cpdata = []
    Cpdata_unit = "cal/(mol*K)"
    nasa_polynomials = []


@dataclass
class MockKineticsDB:
    reactions: dict = field(default_factory=dict)  # key -> {"id": ...}
    rates: dict = field(default_factory=dict)  # id -> rate model (or None)

    def get_reaction_by_reaction(self, reaction):
        return self.reactions.get(reaction.get("label"))

    def get_rate_model(self, reaction_id):
        return self.rates.get(reaction_id, 1)  # default: has a rate


class MockRateModel(Arrhenius):
    """An Arrhenius standing in for an assembled library rate model."""

    pass


class MockKineticsML:
    def __init__(self, covered=()):
        self.covered = set(covered)
        self.called_with = []

    def covers(self, rxn_smiles):
        return rxn_smiles in self.covered

    def predict(self, rxn_smiles, degeneracy=1.0):
        self.called_with.append((rxn_smiles, degeneracy))
        return SimpleKineticsPrediction(A=500.0 * degeneracy, n=0.5, Ea=3000.0)


class SimpleKineticsPrediction:
    def __init__(self, A, n, Ea):
        self.A = A
        self.n = n
        self.Ea = Ea


# ---------------------------------------------------------------------------
# estimate_thermo
# ---------------------------------------------------------------------------


def test_estimate_thermo_library_hit_wins():
    """Library hit wins; the ML estimator is never called."""
    db = MockThermoDB(entries={"ethane": MockThermoEntry(label="ethane")})
    ml = MockML(thermo=MockThermoML(covered=("CC",)))
    counts = EstimationCounts()

    result = estimate_thermo(
        {"label": "ethane", "smiles": "CC"}, db, ml, counts
    )

    # SI conversion: 1.0 kcal/mol -> J/mol, 10.0 cal/(mol*K) -> J/(mol*K)
    assert result.Hf298 == pytest.approx(1.0 * KCAL, rel=1e-12)
    assert result.S298 == pytest.approx(10.0 * Cal, rel=1e-12)
    assert result.Cp_model is not None
    assert counts.library_hits == 1
    assert counts.ml_hits == 0
    assert counts.coverage_errors == 0
    assert counts.total == 1
    assert ml.thermo.called_with == []  # ML never touched


def test_estimate_thermo_library_hit_cp_is_si():
    """The library Cp grid is converted cal -> J and evaluates at any T."""
    db = MockThermoDB(entries={"ethane": MockThermoEntry(label="ethane")})
    ml = MockML(thermo=MockThermoML())
    counts = EstimationCounts()

    result = estimate_thermo({"label": "ethane", "smiles": "CC"}, db, ml, counts)

    # Cp(300) must be the stored grid value in J/(mol*K): 30 cal * 4.184,
    # within the Wilhoit-fit tolerance (the fit max residual is ~0.07 J).
    assert result.Cp_model.get_heat_capacity(300.0) == pytest.approx(
        30.0 * Cal, rel=1e-3
    )
    # and Cp(1500) the top of the grid
    assert result.Cp_model.get_heat_capacity(1500.0) == pytest.approx(
        130.0 * Cal, rel=1e-3
    )


def test_estimate_thermo_ml_hit_when_no_library():
    """No library entry -> the ML estimator is used (exact values pass through)."""
    db = MockThermoDB(entries={})
    ml = MockML(thermo=MockThermoML(covered=("CC",)))
    counts = EstimationCounts()

    result = estimate_thermo({"label": "ghost", "smiles": "CC"}, db, ml, counts)

    assert result.Hf298 == 123.0
    assert result.S298 == 45.0
    assert result.uncertainties["source"] == "mock-ml"
    assert counts.library_hits == 0
    assert counts.ml_hits == 1
    assert counts.coverage_errors == 0
    assert counts.total == 1
    assert ml.thermo.called_with == ["CC"]


def test_estimate_thermo_coverage_error_is_not_swallowed():
    """Neither library nor ML covers it -> MLCoverageError propagates."""
    db = MockThermoDB(entries={})
    ml = MockML(thermo=MockThermoML(covered=("CCC",)))  # does NOT cover CC
    counts = EstimationCounts()

    with pytest.raises(MLCoverageError):
        estimate_thermo({"label": "ghost", "smiles": "CC"}, db, ml, counts)

    assert counts.library_hits == 0
    assert counts.ml_hits == 0
    assert counts.coverage_errors == 1
    assert counts.total == 1
    assert ml.thermo.called_with == []  # predict never reached


def test_estimate_thermo_no_ml_configured_is_coverage_error():
    """ml.thermo is None -> MLCoverageError (no third branch)."""
    db = MockThermoDB(entries={})
    ml = MockML(thermo=None)
    counts = EstimationCounts()

    with pytest.raises(MLCoverageError):
        estimate_thermo({"label": "ghost", "smiles": "CC"}, db, ml, counts)

    assert counts.coverage_errors == 1
    assert counts.ml_hits == 0


def test_estimate_thermo_no_structure_is_coverage_error():
    """No SMILES/adjlist to hand to the ML -> MLCoverageError."""
    db = MockThermoDB(entries={})
    ml = MockML(thermo=MockThermoML(covered=("CC",)))
    counts = EstimationCounts()

    with pytest.raises(MLCoverageError):
        estimate_thermo({"label": "ghost"}, db, ml, counts)

    assert counts.coverage_errors == 1


def test_estimate_thermo_library_row_without_model_is_gap():
    """A library row with no H298/S298/Cp model at all is a coverage gap, not a hit."""
    db = MockThermoDB(entries={"empty": MockNoModelEntry()})
    ml = MockML(thermo=MockThermoML())
    counts = EstimationCounts()

    with pytest.raises(MLCoverageError):
        estimate_thermo({"label": "empty", "smiles": "CC"}, db, ml, counts)

    assert counts.library_hits == 0
    assert counts.coverage_errors == 1


def test_estimate_thermo_nasa_only_entry_uses_nasa_model():
    """NULL H298/S298 (NaN in rmgdb) + NASA polynomials -> the NASA model
    supplies the values (the N2 case in primaryThermoLibrary)."""
    entry = MockThermoEntry(
        label="N2",
        H298=float("nan"),
        S298=float("nan"),
        Tdata=[],
        Cpdata=[],
        nasa_polynomials=[
            # primaryThermoLibrary N2, 200-1000 K segment
            {
                "coeffs": [3.53101, -0.000123661, -5.02999e-07, 2.43531e-09,
                           -1.40881e-12, -1046.98, 2.96747],
                "Tmin": 200.0,
                "Tmax": 1000.0,
            }
        ],
    )
    db = MockThermoDB(entries={"N2": entry})
    ml = MockML(thermo=MockThermoML())
    counts = EstimationCounts()

    result = estimate_thermo({"label": "N2", "smiles": "N#N"}, db, ml, counts)

    # N2 is the reference element: Hf298 ~ 0, S298 ~ 191.6 J/(mol*K)
    assert result.Hf298 == pytest.approx(0.0, abs=100.0)
    assert result.S298 == pytest.approx(191.6, abs=2.0)
    assert counts.library_hits == 1
    assert counts.coverage_errors == 0


def test_estimate_thermo_counts_reflect_split_exactly():
    """Mixed outcomes: the counts object reflects the split exactly."""
    db = MockThermoDB(entries={"lib": MockThermoEntry(label="lib")})
    ml = MockML(thermo=MockThermoML(covered=("CC",)))
    counts = EstimationCounts()

    estimate_thermo({"label": "lib", "smiles": "CC"}, db, ml, counts)      # library
    estimate_thermo({"label": "ml", "smiles": "CC"}, db, ml, counts)       # ML
    no_ml = MockML(thermo=MockThermoML(covered=("CCC",)))
    with pytest.raises(MLCoverageError):
        estimate_thermo({"label": "err", "smiles": "CC"}, db, no_ml, counts)

    assert counts.library_hits == 1
    assert counts.ml_hits == 1
    assert counts.coverage_errors == 1
    assert counts.total == 3  # 3 attempts, 3 outcomes, nothing swallowed
    assert counts.as_dict() == {
        "library_hits": 1, "ml_hits": 1, "coverage_errors": 1, "total": 3,
    }


# ---------------------------------------------------------------------------
# estimate_kinetics
# ---------------------------------------------------------------------------


def test_estimate_kinetics_library_hit():
    """Library hit wins: the assembled rate model is returned, ML untouched."""
    library_rate = MockRateModel(A=1.0e6, n=1.5, Ea=20000.0)
    db = MockKineticsDB(
        reactions={"rxn1": {"id": 1}},
        rates={1: library_rate},
    )
    ml = MockML(kinetics=MockKineticsML(covered=("A>>B",)))
    counts = EstimationCounts()

    rate, deg = estimate_kinetics(
        {"label": "rxn1", "reactants": [{"smiles": "A"}],
         "products": [{"smiles": "B"}]},
        db, ml, counts,
    )

    assert rate is library_rate
    assert deg == 1.0
    assert counts.library_hits == 1
    assert counts.ml_hits == 0
    assert counts.coverage_errors == 0
    assert ml.kinetics.called_with == []  # ML never touched


def test_estimate_kinetics_ml_hit_when_no_library():
    """No library match -> ML prediction becomes an Arrhenius (exact values)."""
    db = MockKineticsDB(reactions={})
    ml = MockML(kinetics=MockKineticsML(covered=("A>>B",)))
    counts = EstimationCounts()

    rate, deg = estimate_kinetics(
        {"label": "ghost", "reactants": [{"smiles": "A"}],
         "products": [{"smiles": "B"}]},
        db, ml, counts,
    )

    assert isinstance(rate, Arrhenius)
    assert rate.A == pytest.approx(500.0)
    assert rate.n == pytest.approx(0.5)
    assert rate.Ea == pytest.approx(3000.0)
    assert deg == 1.0
    assert counts.library_hits == 0
    assert counts.ml_hits == 1
    assert counts.coverage_errors == 0
    assert ml.kinetics.called_with == [("A>>B", 1.0)]


def test_estimate_kinetics_degeneracy_boundary_conversion():
    """The degeneracy argument is passed through to the ML (A = 10^pred * deg)."""
    db = MockKineticsDB(reactions={})
    ml = MockML(kinetics=MockKineticsML(covered=("A>>B",)))
    counts = EstimationCounts()

    rate, deg = estimate_kinetics(
        {"label": "ghost", "reactants": [{"smiles": "A"}],
         "products": [{"smiles": "B"}]},
        db, ml, counts, degeneracy=3.0,
    )

    assert rate.A == pytest.approx(500.0 * 3.0)
    assert deg == pytest.approx(3.0)
    assert ml.kinetics.called_with == [("A>>B", 3.0)]


def test_estimate_kinetics_gap_reaction_falls_to_ml():
    """A matched library reaction whose rate is None (rmgdb gap) -> ML."""
    db = MockKineticsDB(
        reactions={"gap": {"id": 7}},
        rates={7: None},  # stored, but not assemblable (documented gap)
    )
    ml = MockML(kinetics=MockKineticsML(covered=("A>>B",)))
    counts = EstimationCounts()

    rate, deg = estimate_kinetics(
        {"label": "gap", "reactants": [{"smiles": "A"}],
         "products": [{"smiles": "B"}]},
        db, ml, counts,
    )

    assert isinstance(rate, Arrhenius)
    assert rate.A == pytest.approx(500.0)
    assert counts.library_hits == 0
    assert counts.ml_hits == 1
    assert counts.coverage_errors == 0


def test_estimate_kinetics_coverage_error_is_not_swallowed():
    """Neither library nor ML covers it -> MLCoverageError propagates."""
    db = MockKineticsDB(reactions={})
    ml = MockML(kinetics=MockKineticsML(covered=("C>>D",)))  # not A>>B
    counts = EstimationCounts()

    with pytest.raises(MLCoverageError):
        estimate_kinetics(
            {"label": "ghost", "reactants": [{"smiles": "A"}],
             "products": [{"smiles": "B"}]},
            db, ml, counts,
        )

    assert counts.library_hits == 0
    assert counts.ml_hits == 0
    assert counts.coverage_errors == 1
    assert counts.total == 1
    assert ml.kinetics.called_with == []


def test_estimate_kinetics_no_ml_is_coverage_error():
    """ml.kinetics is None -> MLCoverageError (no third branch)."""
    db = MockKineticsDB(reactions={})
    ml = MockML(kinetics=None)
    counts = EstimationCounts()

    with pytest.raises(MLCoverageError):
        estimate_kinetics(
            {"label": "ghost", "reactants": [{"smiles": "A"}],
             "products": [{"smiles": "B"}]},
            db, ml, counts,
        )

    assert counts.coverage_errors == 1


# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------


def test_reaction_to_smiles_uses_explicit_reaction_smiles():
    """reaction_smiles is authoritative (the '>>' RIGR format)."""
    reaction = {
        "reaction_smiles": "A:1>>B:1",
        "reactants": [{"smiles": "WRONG"}],
        "products": [{"smiles": "ALSO_WRONG"}],
    }
    assert reaction_to_smiles(reaction) == "A:1>>B:1"


def test_reaction_to_smiles_built_from_species():
    reaction = {
        "reactants": [{"smiles": "A"}, {"smiles": "B"}],
        "products": [{"smiles": "C"}],
    }
    assert reaction_to_smiles(reaction) == "A.B>>C"


def test_reaction_to_smiles_empty_when_incomplete():
    assert reaction_to_smiles({"reactants": [{"smiles": "A"}]}) == ""
    assert reaction_to_smiles({}) == ""


def test_species_to_smiles_prefers_smiles():
    assert species_to_smiles({"smiles": "CC", "label": "ethane"}) == "CC"


# ---------------------------------------------------------------------------
# The lookup_kinetics adapter (job-02 stub -> resolver)
# ---------------------------------------------------------------------------


def test_lookup_kinetics_wires_the_resolver():
    """lookup_kinetics routes through estimate_kinetics (the TODO is gone)."""
    from rmgpu.data.kinetics import lookup_kinetics

    library_rate = MockRateModel(A=2.0e6, n=0.0, Ea=5000.0)
    db = MockKineticsDB(reactions={"rxn1": {"id": 1}}, rates={1: library_rate})
    ml = MockML(kinetics=MockKineticsML(covered=("A>>B",)))
    counts = EstimationCounts()

    res = lookup_kinetics(
        {"label": "rxn1", "reactants": [{"smiles": "A"}],
         "products": [{"smiles": "B"}]},
        db, ml, counts=counts,
    )
    assert res.found is True
    assert res.source == "library"
    assert res.rate_model is library_rate
    assert counts.library_hits == 1

    res2 = lookup_kinetics(
        {"label": "ghost", "reactants": [{"smiles": "A"}],
         "products": [{"smiles": "B"}]},
        db, ml, counts=counts,
    )
    assert res2.found is True
    assert res2.source == "ml"
    assert isinstance(res2.rate_model, Arrhenius)
    assert counts.ml_hits == 1

    res3 = lookup_kinetics(
        {"label": "none", "reactants": [{"smiles": "X"}],
         "products": [{"smiles": "Y"}]},
        db, ml, counts=counts,
    )
    assert res3.found is False
    assert res3.source == ""
    assert counts.coverage_errors == 1
    assert counts.total == 3
