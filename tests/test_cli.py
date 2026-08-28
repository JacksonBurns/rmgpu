import os
import pytest
from click.testing import CliRunner

from rmgpu.cli import main
from rmgpu.schemas.input import Input

@pytest.fixture
def minimal_yaml(tmp_path):
    content = """
rmgpu: "1.0"
database:
  thermo_libraries:
    - primaryThermoLibrary
  reaction_libraries:
    - primaryH2O2
  kinetics_families: default
  kinetics_estimator: ml
species:
  - label: CH4
    reactive: true
    structure:
      smiles: C
  - label: O2
    reactive: true
    structure:
      smiles: "[O]=O"
reactors:
  - type: simple
    temperature: 1000 K
    pressure: 1 bar
    initial_mole_fractions:
      CH4: 0.1
      O2: 0.4
    termination:
      time: 10 ms
      conversion:
        species: CH4
        value: 0.9
simulator:
  atol: 1.0e-16
  rtol: 1.0e-8
model:
  tolerance_move_to_core: 0.1
  tolerance_keep_in_edge: 0.0
  tolerance_interrupt_simulation: 1.0
  maximum_edge_species: 100000
options:
  save_profiles: false
  save_plots: false
  save_edge: false
  units: si
"""
    path = tmp_path / "minimal.yaml"
    path.write_text(content)
    return str(path)

@pytest.fixture
def bad_yaml(tmp_path):
    # Three problems:
    # 1. rmgpu version is missing (required)
    # 2. species[0].label is invalid (contains +)
    # 3. reactors[0].temperature is not a quantity string
    # 4. simulator.atol is not a number
    content = """
database:
  thermo_libraries:
    - primaryThermoLibrary
species:
  - label: CH4+
    reactive: true
    structure:
      smiles: C
  - label: O2
    reactive: true
    structure:
      smiles: "[O]=O"
reactors:
  - type: simple
    temperature: not_a_quantity
    pressure: 1 bar
    initial_mole_fractions:
      CH4: 0.1
      O2: 0.4
simulator:
  atol: not_a_number
  rtol: 1.0e-8
"""
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    return str(path)

def test_run_prints_valid_yaml(minimal_yaml):
    runner = CliRunner()
    result = runner.invoke(main, ["run", minimal_yaml])
    assert result.exit_code == 0, result.output
    # Output should be valid YAML with the resolved document
    assert "rmgpu: '1.0'" in result.output
    assert "database:" in result.output
    assert "CH4" in result.output

def test_validate_passes(minimal_yaml):
    runner = CliRunner()
    result = runner.invoke(main, ["validate", minimal_yaml])
    assert result.exit_code == 0, result.output
    assert "Validation passed." in result.output

def test_validate_reports_all_problems(bad_yaml):
    runner = CliRunner()
    result = runner.invoke(main, ["validate", bad_yaml])
    assert result.exit_code == 1
    # Should report at least one quantity error and one validation error
    output = result.output
    assert "not_a_quantity" in output
    assert "Validation error" in output

def test_schema_exports_and_roundtrips(tmp_path, minimal_yaml):
    schema_path = tmp_path / "schema.yaml"
    runner = CliRunner()
    result = runner.invoke(main, ["schema", "--out", str(schema_path)])
    assert result.exit_code == 0, result.output
    assert schema_path.exists()

    # Validate the minimal yaml against the exported schema using jsonschema
    import yaml
    import json
    with open(schema_path) as f:
        schema = yaml.safe_load(f)
    with open(minimal_yaml) as f:
        instance = yaml.safe_load(f)
    from jsonschema import validate as json_validate
    json_validate(instance=instance, schema=schema)

def test_version():
    runner = CliRunner()
    result = runner.invoke(main, ["version"])
    assert result.exit_code == 0
    from rmgpu import __version__
    assert result.output.strip() == __version__

def test_import_roundtrip(tmp_path):
    """job-03/step-05: `rmgpu import` is implemented (lossless legacy import)."""
    src = tmp_path / "input.py"
    src.write_text(
        "database(thermoLibraries=['primaryThermoLibrary'], "
        "reactionLibraries=['primaryReactions'])\n"
        "species(label='CH4', structure=SMILES('C'))\n"
        "simpleReactor(temperature=(800, 'K'), pressure=(1.0, 'bar'), "
        "initialMoleFractions={'CH4': 1.0}, terminationTime=(100, 'us'))\n"
        "simulator(atol=1e-16, rtol=1e-8)\n"
        "model(toleranceMoveToCore=0.1)\n"
    )
    out = tmp_path / "out.yaml"
    runner = CliRunner()
    result = runner.invoke(main, ["import", str(src), "--to", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    from rmgpu.schemas.input import load_input
    doc = load_input(str(out))
    assert doc.rmgpu == "1.0"
    assert doc.species[0].label == "CH4"

def test_export_diff_inspect_not_implemented(tmp_path):
    dummy = tmp_path / "dummy.txt"
    dummy.write_text("dummy")
    runner = CliRunner()
    for command in ["export", "diff", "inspect"]:
        result = runner.invoke(main, [command, str(dummy)])
        assert result.exit_code == 0
        assert "not yet implemented" in result.output
