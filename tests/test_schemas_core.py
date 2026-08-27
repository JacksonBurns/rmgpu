"""Tests for rmgpu.schemas.input core schema blocks."""

import textwrap
from pathlib import Path

import pytest

from rmgpu.schemas.input import (
    DatabaseBlock,
    ForbiddenEntry,
    Input,
    Species,
    StructureValue,
    resolve_extends,
)
from rmgpu.units import Quantity


class TestQuantity:
    def test_quantity_from_dict(self):
        q = Quantity(100, "K")
        assert q._value == 100

    def test_quantity_from_string(self):
        q = Quantity(100.5, "bar")
        assert q._value == 100.5

    def test_quantity_to_si(self):
        q = Quantity(1, "kPa")
        assert q.to_si() == 1000.0


class TestStructureValue:
    def test_valid_smiles(self):
        sv = StructureValue(smiles="CC")
        assert sv.value == "CC"

    def test_valid_adjlist(self):
        sv = StructureValue(adjlist="1 C u0 {2 S}")
        assert sv.value == "1 C u0 {2 S}"

    def test_invalid_missing_both(self):
        with pytest.raises(ValueError):
            StructureValue()


class TestDatabaseBlock:
    def test_defaults(self):
        db = DatabaseBlock()
        assert db.kinetics_estimator == "ml"
        assert db.kinetics_families == "default"
        assert db.transport_libraries == "auto"

    def test_custom_values(self):
        db = DatabaseBlock(
            thermo_libraries=["primaryThermoLibrary"],
            kinetics_families=["H_Abstraction"],
            kinetics_depositories=["training"],
        )
        assert db.thermo_libraries == ["primaryThermoLibrary"]
        assert db.kinetics_families == ["H_Abstraction"]
        assert db.kinetics_depositories == ["training"]


class TestSpecies:
    def test_valid_species(self):
        sp = Species(label="ethane", structure={"smiles": "CC"})
        assert sp.label == "ethane"
        assert sp.reactive is True

    def test_invalid_label_with_plus(self):
        with pytest.raises(ValueError):
            Species(label="CH3+", structure={"smiles": "C"})

    def test_invalid_label_with_plus2(self):
        with pytest.raises(ValueError):
            Species(label="A+B", structure={"smiles": "C"})


class TestForbiddenEntry:
    def test_valid_entry(self):
        entry = ForbiddenEntry(structure="C=[C]=[C]", reason="cumulenes unsupported")
        assert entry.structure == "C=[C]=[C]"


class TestInput:
    def test_valid_input(self):
        doc = {
            "rmgpu": "1.0",
            "database": {"thermo_libraries": ["primaryThermoLibrary"]},
            "species": [{"label": "ethane", "structure": {"smiles": "CC"}}],
            "forbidden": [{"structure": "C=[C]=[C]", "reason": "unsupported"}],
        }
        inp = Input(**doc)
        assert inp.rmgpu == "1.0"

    def test_invalid_version(self):
        with pytest.raises(ValueError):
            Input(rmgpu="2.0")

    def test_missing_version(self):
        with pytest.raises(ValueError):
            Input(database={})


class TestDumpJsonSchema:
    def test_dump_json_schema(self):
        schema = Input.dump_json_schema()
        assert "rmgpu" in schema["properties"]
        assert schema["properties"]["rmgpu"]["type"] == "string"
        assert schema["properties"]["rmgpu"]["pattern"] == r"^\d+\.\d+$"


class TestResolveExtends:
    def test_no_extends(self):
        doc = {"rmgpu": "1.0"}
        result = resolve_extends(doc)
        assert result == doc

    def test_extends_with_extends_in_extended(self, tmp_path):
        base = tmp_path / "base.yaml"
        ext = tmp_path / "ext.yaml"
        base.write_text(textwrap.dedent("""
            rmgpu: 1.0
            database:
              thermo_libraries:
                - primaryThermoLibrary
        """))
        ext.write_text(textwrap.dedent("""
            rmgpu: 1.0
            extends: ./other.yaml
        """))
        doc = {"rmgpu": "1.0", "extends": str(ext)}
        with pytest.raises(ValueError):
            resolve_extends(doc)

    def test_extends_missing_file(self, tmp_path):
        doc = {"rmgpu": "1.0", "extends": "/nonexistent/file.yaml"}
        with pytest.raises((OSError, FileNotFoundError)):
            resolve_extends(doc)

    def test_two_level_chain(self, tmp_path):
        base = tmp_path / "base.yaml"
        ext1 = tmp_path / "ext1.yaml"
        ext2 = tmp_path / "ext2.yaml"
        base.write_text(textwrap.dedent("""
            rmgpu: 1.0
            database:
              thermo_libraries:
                - primaryThermoLibrary
            species:
              - label: ethane
                structure:
                  smiles: "CC"
        """))
        ext1.write_text(textwrap.dedent("""
            rmgpu: 1.0
            species:
              - label: methane
                structure:
                  smiles: "C"
        """))
        ext2.write_text(textwrap.dedent("""
            rmgpu: 1.0
            forbidden:
              - structure: "C=[C]=[C]"
                reason: "unsupported"
        """))
        doc = {"rmgpu": "1.0", "extends": [str(ext1), str(ext2)]}
        result = resolve_extends(doc)
        assert "species" in result
        assert "forbidden" in result
