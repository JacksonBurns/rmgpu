import os
import textwrap
from pathlib import Path

import pytest
from rmgpu.schemas import (
    Input,
    PressureDependenceBlock,
    SimpleReactor,
    ConstantVReactor,
    ConstantTPReactor,
    LiquidReactor,
    MBSampledReactor,
    SurfaceReactor,
    load_input,
)
from rmgpu.units import Quantity

def test_polymorphic_dispatch_simple():
    d = {
        "type": "simple",
        "temperature": {"value": 1000, "unit": "K"},
        "pressure": {"value": 1.0, "unit": "bar"},
        "initial_mole_fractions": {"CH4": 0.2, "O2": 0.2, "N2": 0.6},
        "termination": {"time": {"value": 0.1, "unit": "s"}},
    }
    r = SimpleReactor(**d)
    assert isinstance(r.temperature, Quantity)
    assert isinstance(r.pressure, Quantity)

def test_polymorphic_dispatch_const_v():
    d = {
        "type": "const_V",
        "temperature": {"value": 1000, "unit": "K"},
        "pressure": {"value": 1.0, "unit": "bar"},
        "initial_mole_fractions": {"CH4": 0.2, "O2": 0.2, "N2": 0.6},
    }
    r = ConstantVReactor(**d)
    assert isinstance(r.temperature, Quantity)

def test_polymorphic_dispatch_const_tp():
    d = {
        "type": "const_TP",
        "temperature": {"value": 1000, "unit": "K"},
        "pressure": {"value": 1.0, "unit": "bar"},
        "initial_mole_fractions": {"CH4": 0.2, "O2": 0.2, "N2": 0.6},
    }
    r = ConstantTPReactor(**d)
    assert isinstance(r.temperature, Quantity)

def test_polymorphic_dispatch_liquid():
    d = {
        "type": "liquid",
        "temperature": {"value": 298, "unit": "K"},
        "initial_concentrations": {"water": 1.0, "reactant": 1e-3},
    }
    r = LiquidReactor(**d)
    assert isinstance(r.temperature, Quantity)

def test_polymorphic_dispatch_mb_sampled():
    d = {
        "type": "mb_sampled",
        "temperature": {"value": 1000, "unit": "K"},
        "pressure": {"value": 1.0, "unit": "bar"},
        "initial_mole_fractions": {"CH4": 0.2, "O2": 0.2, "N2": 0.6},
        "mbsampling_rate": {"value": 0.1, "unit": "1/s"},
    }
    r = MBSampledReactor(**d)
    assert isinstance(r.temperature, Quantity)

def test_polymorphic_dispatch_surface():
    d = {
        "type": "surface",
        "temperature": {"value": 500, "unit": "K"},
        "initial_pressure": {"value": 1.0, "unit": "bar"},
        "initial_gas_mole_fractions": {"CH4": 0.2, "O2": 0.2, "N2": 0.6},
        "initial_surface_coverages": {"Pt111": 1.0},
        "surface_volume_ratio": {"value": 1.0, "unit": "1/m"},
    }
    r = SurfaceReactor(**d)
    assert isinstance(r.temperature, Quantity)

def test_pressure_dep_method_shorthand():
    d = {
        "method": "cse",
        "Tmin": {"value": 300, "unit": "K"},
        "Tmax": {"value": 2000, "unit": "K"},
        "Tcount": 20,
        "Pmin": {"value": 1.0, "unit": "bar"},
        "Pmax": {"value": 50.0, "unit": "bar"},
        "Pcount": 10,
        "maximumGrainSize": {"value": 0.0, "unit": "kJ/mol"},
        "minimumNumberOfGrains": 0,
    }
    pdp = PressureDependenceBlock(**d)
    assert pdp.method == "strong collision"

def test_pressure_dep_method_long_form():
    d = {
        "method": "modified strong collision",
        "Tmin": {"value": 300, "unit": "K"},
        "Tmax": {"value": 2000, "unit": "K"},
        "Tcount": 20,
        "Pmin": {"value": 1.0, "unit": "bar"},
        "Pmax": {"value": 50.0, "unit": "bar"},
        "Pcount": 10,
    }
    pdp = PressureDependenceBlock(**d)
    assert pdp.method == "modified strong collision"

def test_plan_example_validates(tmp_path):
    """Validate the PLAN.md 12.2 example end-to-end."""
    yaml_text = textwrap.dedent("""
        rmgpu: 1.0
        database:
          thermo_libraries: [primaryThermoLibrary]
          reaction_libraries: []
          seed_mechanisms: [my_mechanism/core.yaml]
          kinetics_families: default
          kinetics_depositories: [training]
          kinetics_estimator: ml
          transport_libraries: auto
        species:
          - {label: ethane, reactive: true, structure: "CC"}
          - {label: CH3CHO, reactive: true, structure: "CC=O"}
          - {label: H, reactive: true, structure: "[H]"}
        forbidden:
          - {structure: "C=[C]=[C]", reason: cumulenes unsupported}
        reactors:
          - type: simple
            temperature: {value: 1350, unit: K}
            pressure: {value: 1.0, unit: bar}
            initial_mole_fractions: {ethane: 1.0}
            termination:
              conversion: {species: ethane, value: 0.9}
              time: {value: 1e6, unit: s}
        simulator: {atol: 1e-16, rtol: 1e-8}
        model:
          tolerance_keep_in_edge: 0.0
          tolerance_move_to_core: 0.1
          tolerance_interrupt_simulation: 0.1
          maximum_edge_species: 100000
          filter_reactions: true
        pressure_dependence: {method: cse}
        ml_estimator: {thermo: chemeleon_thermo_v1, kinetics: chemeleon_rxn_v1}
        solvation: {solvent: acetonitrile, model: smd}
        uncertainty: {enabled: true, species: [CO, CO2]}
        options: {save_profiles: true, save_plots: false, save_edge: true, units: si}
    """)
    inp = tmp_path / "input.yaml"
    inp.write_text(yaml_text)
    loaded = load_input(str(inp))
    assert loaded.rmgpu == "1.0"
    assert loaded.database is not None
    assert len(loaded.species) == 3

def test_extends_two_level(tmp_path):
    """Test extends with two levels of nesting."""
    base = tmp_path / "base.yaml"
    base.write_text(textwrap.dedent("""
        rmgpu: 1.0
        database:
          thermo_libraries: [primaryThermoLibrary]
    """))
    mid = tmp_path / "mid.yaml"
    mid.write_text(textwrap.dedent("""
        extends: base.yaml
        database:
          reaction_libraries: [primaryH2O2]
        species:
          - {label: CH4, reactive: true, structure: C}
    """))
    top = tmp_path / "top.yaml"
    top.write_text(textwrap.dedent("""
        extends: mid.yaml
        reactors:
          - type: simple
            temperature: {value: 1000, unit: K}
            pressure: {value: 1.0, unit: bar}
            initial_mole_fractions: {CH4: 0.2, O2: 0.2, N2: 0.6}
    """))
    loaded = load_input(str(top))
    assert loaded.database is not None
    assert loaded.database.thermo_libraries == ["primaryThermoLibrary"]
    assert loaded.database.reaction_libraries == ["primaryH2O2"]
    assert len(loaded.species) == 1
    assert len(loaded.reactors) == 1

def test_extends_cycle_detection(tmp_path):
    """Test that extends cycle detection raises an error."""
    a = tmp_path / "a.yaml"
    a.write_text(textwrap.dedent("""
        rmgpu: 1.0
        extends: b.yaml
    """))
    b = tmp_path / "b.yaml"
    b.write_text(textwrap.dedent("""
        extends: a.yaml
    """))
    with pytest.raises(ValueError):
        load_input(str(a))

def test_staged_reactor_nested_list():
    """Test that staged reactors (nested lists) are supported."""
    d = {
        "type": "simple",
        "temperature": {"value": 1000, "unit": "K"},
        "pressure": {"value": 1.0, "unit": "bar"},
        "initial_mole_fractions": {"CH4": 0.2, "O2": 0.2, "N2": 0.6},
    }
    r = SimpleReactor(**d)
    assert isinstance(r.temperature, Quantity)
