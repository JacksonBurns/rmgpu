"""Tests for job-03/step-02: reactors, remaining blocks, extends."""
import os
import pytest
from rmgpu.schemas.input import (
    Input,
    SimpleReactor,
    ConstantVReactor,
    ConstantTPReactor,
    LiquidReactor,
    MBSampledReactor,
    SurfaceReactor,
    StagedReactor,
    LiquidStagedReactor,
    ConstantVStagedReactor,
    PressureStagedReactor,
    SimulatorBlock,
    ModelBlock,
    PressureDependenceBlock,
    MLEstimatorBlock,
    SolvationBlock,
    UncertaintyBlock,
    OptionsBlock,
    load_input,
    resolve_extends,
)
from rmgpu.units import Quantity

# --- Polymorphic dispatch tests ---

def test_simple_reactor_dispatch():
    reactor = SimpleReactor(
        temperature="300 K",
        pressure="1 bar",
        initial_mole_fractions={"H2": 0.5, "CH4": 0.5},
    )
    assert reactor.type == "simple"

def test_const_v_reactor_dispatch():
    reactor = ConstantVReactor(
        temperature="500 K",
        pressure="10 bar",
        initial_mole_fractions={"C3H8": 1.0},
    )
    assert reactor.type == "const_V"

def test_const_tp_reactor_dispatch():
    reactor = ConstantTPReactor(
        temperature="1000 K",
        pressure="1 bar",
        initial_mole_fractions={"CH4": 0.5, "O2": 0.5},
    )
    assert reactor.type == "const_TP"

def test_liquid_reactor_dispatch():
    reactor = LiquidReactor(
        temperature="298 K",
        initial_concentrations={"water": 55.5},
    )
    assert reactor.type == "liquid"

def test_mb_sampled_reactor_dispatch():
    reactor = MBSampledReactor(
        temperature="300 K",
        pressure="1 bar",
        initial_mole_fractions={"A": 1.0},
        mbsampling_rate="1000 /s",
    )
    assert reactor.type == "mb_sampled"

def test_surface_reactor_dispatch():
    reactor = SurfaceReactor(
        temperature="500 K",
        initial_pressure="1 bar",
        initial_gas_mole_fractions={"H2": 1.0},
        initial_surface_coverages={"H*": 0.1},
        surface_volume_ratio="1e-10 m3",
    )
    assert reactor.type == "surface"

# --- Staged reactor tests ---

def test_staged_reactor():
    reactor = StagedReactor(
        temperature="500 K",
        pressure="1 bar",
        initial_mole_fractions={"A": 0.5, "B": 0.5},
        staged_initial_mole_fractions=[
            {"A": 0.5, "B": 0.5},
            {"A": 0.7, "B": 0.3},
        ],
    )
    assert reactor.type == "staged"
    assert len(reactor.staged_initial_mole_fractions) == 2

def test_staged_reactor_requires_stages():
    with pytest.raises(Exception):
        StagedReactor(
            temperature="500 K",
            pressure="1 bar",
            initial_mole_fractions={"A": 1.0},
            staged_initial_mole_fractions=[],
        )

def test_liquid_staged_reactor():
    reactor = LiquidStagedReactor(
        temperature="298 K",
        initial_concentrations={"water": 55.5},
        staged_temperatures=["298 K", "310 K"],
    )
    assert reactor.type == "staged"
    assert len(reactor.staged_temperatures) == 2

def test_const_v_staged_reactor():
    reactor = ConstantVStagedReactor(
        temperature="500 K",
        pressure="1 bar",
        initial_mole_fractions={"A": 1.0},
        staged_temperatures=["500 K", "600 K"],
    )
    assert reactor.type == "staged"

def test_pressure_staged_reactor():
    reactor = PressureStagedReactor(
        temperature="500 K",
        pressure="1 bar",
        initial_mole_fractions={"A": 1.0},
        staged_temperatures=["500 K", "600 K"],
        staged_pressures=["1 bar", "2 bar"],
    )
    assert reactor.type == "staged"

# --- Pressure dependence method normalization ---

def test_method_shorthand_cse():
    pd = PressureDependenceBlock(method="cse")
    assert pd.method == "strong collision"

def test_method_shorthand_masc():
    pd = PressureDependenceBlock(method="masc")
    assert pd.method == "modified strong collision"

def test_method_shorthand_rs():
    pd = PressureDependenceBlock(method="rs")
    assert pd.method == "randomized strong collision"

def test_method_shorthand_sls():
    pd = PressureDependenceBlock(method="sls")
    assert pd.method == "super logarithmic spacing"

def test_method_long_form_passthrough():
    pd = PressureDependenceBlock(method="modified strong collision")
    assert pd.method == "modified strong collision"

# --- Extends tests ---

def test_extends_two_level(tmp_path):
    """Two-level extends chain with cycle detection."""
    base = tmp_path / "base.yaml"
    base.write_text("""
rmgpu: "1.0"
database:
  thermo_libraries: ["primaryThermoLibrary"]
species:
  - label: H2
    structure: "H[H]"
""")
    mid = tmp_path / "mid.yaml"
    mid.write_text("""
rmgpu: "1.0"
extends: base.yaml
reactors:
  - type: simple
    temperature: "300 K"
    pressure: "1 bar"
    initial_mole_fractions:
      H2: 1.0
""")
    final = tmp_path / "final.yaml"
    final.write_text("""
rmgpu: "1.0"
extends: mid.yaml
model:
  tolerance_move_to_core: 0.2
""")
    resolved = resolve_extends({"rmgpu": "1.0"}, str(tmp_path))
    # Actually test load_input
    inp = load_input(str(final))
    assert inp.rmgpu == "1.0"
    assert inp.model is not None
    assert inp.model.tolerance_move_to_core == 0.2
    assert inp.database is not None
    assert inp.database.thermo_libraries == ["primaryThermoLibrary"]
    assert inp.reactors is not None

def test_extends_cycle_detection(tmp_path):
    """Cycle in extends chain should raise."""
    a = tmp_path / "a.yaml"
    a.write_text("""
rmgpu: "1.0"
extends: b.yaml
""")
    b = tmp_path / "b.yaml"
    b.write_text("""
rmgpu: "1.0"
extends: a.yaml
""")
    # Load a.yaml which extends b.yaml which extends a.yaml (cycle)
    with pytest.raises(ValueError):
        load_input(str(a))

# --- Block tests ---

def test_simulator_block():
    sim = SimulatorBlock(atol=1e-10, rtol=1e-5)
    assert sim.atol == 1e-10
    assert sim.rtol == 1e-5

def test_model_block():
    m = ModelBlock(tolerance_move_to_core=0.05, maximum_edge_species=500)
    assert m.tolerance_move_to_core == 0.05
    assert m.maximum_edge_species == 500

def test_ml_estimator_block():
    ml = MLEstimatorBlock(thermo="example_thermo.ckpt", kinetics="example_kinetics.ckpt")
    assert ml.thermo == "example_thermo.ckpt"

def test_solvation_block():
    s = SolvationBlock(solvent="water")
    assert s.solvent == "water"
    assert s.model == "smd"

def test_uncertainty_block():
    u = UncertaintyBlock(enabled=True, species=["CH4"])
    assert u.enabled is True
    assert u.species == ["CH4"]

def test_options_block():
    o = OptionsBlock(save_profiles=True, save_plots=True, save_edge=True, units="cgs")
    assert o.save_profiles is True
    assert o.units == "cgs"

# --- PLAN.md 12.2 example end-to-end ---

def test_plan_12_2_example():
    inp = Input(
        rmgpu="1.0",
        database={
            "thermo_libraries": ["primaryThermoLibrary"],
            "kinetics_families": ["H_abstraction", "R_addition"],
            "kinetics_estimator": "ml",
        },
        species=[
            {"label": "CH4", "structure": "C"},
            {"label": "H2", "structure": "H[H]"},
            {"label": "H2O", "structure": "O"},
        ],
        reactors=[
            {
                "type": "simple",
                "temperature": "1000 K",
                "pressure": "1 bar",
                "initial_mole_fractions": {"CH4": 0.5, "H2": 0.5},
            }
        ],
        simulator={"atol": 1e-12, "rtol": 1e-8},
        model={
            "tolerance_move_to_core": 0.2,
            "tolerance_keep_in_edge": 0.001,
            "tolerance_interrupt_simulation": 1.0,
            "maximum_edge_species": 10000,
            "filter_reactions": True,
        },
        pressure_dependence={
            "method": "modified strong collision",
            "Tmin": "300 K",
            "Tmax": "2000 K",
            "Tcount": 20,
            "Pmin": "1 bar",
            "Pmax": "50 bar",
            "Pcount": 10,
        },
        ml_estimator={
            "thermo": "example_thermo_1.0.0",
            "kinetics": "example_kinetics_1.0.0",
        },
        options={"save_profiles": False, "save_edge": True, "units": "si"},
    )
    assert inp.rmgpu == "1.0"
    assert inp.model.tolerance_move_to_core == 0.2
    assert inp.pressure_dependence.method == "modified strong collision"