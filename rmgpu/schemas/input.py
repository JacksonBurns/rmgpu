"""Core input schema blocks for rmgpu YAML input documents."""

import re
from typing import Any, Dict, List, Literal, Optional, Union

import pydantic
from pydantic import Field, field_validator, model_validator

from rmgpu.units import Quantity

__all__ = [
    "Quantity",
    "StructureValue",
    "DatabaseBlock",
    "Species",
    "ForbiddenEntry",
    "Input",
    "resolve_extends",
]

_VERSION_PATTERN = re.compile(r"^\d+\.\d+$")


def _coerce_quantity(v: Any) -> Quantity:
    """Convert a dict, string, or Quantity to a Quantity."""
    if isinstance(v, Quantity):
        return v
    if isinstance(v, str):
        m = re.match(r"^\s*([0-9.eE+-]+)\s*([A-Za-z/\s]+?)\s*$", v)
        if not m:
            raise ValueError(f"Invalid quantity string: {v!r}")
        return Quantity(float(m.group(1)), m.group(2))
    if isinstance(v, dict):
        return Quantity(v["value"], v["unit"])
    raise ValueError(f"Invalid quantity input: {v!r}")


class StructureValue(pydantic.BaseModel):
    """A structure specification: SMILES/InChI string or adjacency list dict."""

    smiles: Optional[str] = None
    adjlist: Optional[str] = None

    @model_validator(mode="after")
    def _validate(self) -> "StructureValue":
        if self.smiles is None and self.adjlist is None:
            raise ValueError("structure must provide smiles or adjlist")
        return self

    @property
    def value(self) -> str:
        return self.smiles or self.adjlist


class DatabaseBlock(pydantic.BaseModel):
    """Database configuration block."""

    thermo_libraries: Optional[List[str]] = None
    reaction_libraries: Optional[List[str]] = None
    seed_mechanisms: Optional[List[str]] = None
    kinetics_families: Union[Literal["default"], List[str]] = "default"
    kinetics_depositories: Optional[Union[Literal["default", "all"], List[str]]] = None
    kinetics_estimator: Literal["ml", "library"] = "ml"
    transport_libraries: Union[Literal["auto"], List[str]] = "auto"


class Species(pydantic.BaseModel):
    """A chemical species declaration."""

    label: str = Field(..., pattern=r"^[^+]+$")
    reactive: bool = True
    structure: StructureValue
    thermo: Optional[Dict[str, Any]] = None
    kinetics: Optional[Dict[str, Any]] = None
    constraints: Optional[List[str]] = None


class ForbiddenEntry(pydantic.BaseModel):
    """A forbidden structure entry."""

    structure: str
    reason: Optional[str] = None


class Input(pydantic.BaseModel):
    """Top-level input document."""

    rmgpu: str = Field(..., pattern=_VERSION_PATTERN)
    extends: Optional[Union[str, List[str]]] = None
    database: Optional[DatabaseBlock] = None
    species: Optional[List[Species]] = None
    forbidden: Optional[List[ForbiddenEntry]] = None

    @field_validator("rmgpu")
    @classmethod
    def _validate_version(cls, v: str) -> str:
        if v != "1.0":
            raise ValueError(f"Unsupported rmgpu version: {v}")
        return v

    @classmethod
    def dump_json_schema(cls) -> Dict[str, Any]:
        return cls.model_json_schema()


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_one(base_doc: Dict[str, Any], ext_path: str) -> Dict[str, Any]:
    """Resolve a single extends path, applying the extends semantics."""
    ext_doc = _load_yaml(ext_path)
    if "extends" in ext_doc:
        raise ValueError("Cannot have extends in an extended document")
    if "rmgpu" in ext_doc:
        ext_version = str(ext_doc["rmgpu"])
        base_version = str(base_doc.get("rmgpu", "1.0"))
        if ext_version != base_version:
            raise ValueError("Version mismatch in extends document")
    return ext_doc


def resolve_extends(base_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve extends chains into a single document."""
    if "extends" not in base_doc or not base_doc["extends"]:
        return base_doc
    extends = base_doc["extends"]
    if isinstance(extends, str):
        extends = [extends]
    result = dict(base_doc)
    result.pop("extends", None)
    for ext_path in extends:
        ext_doc = _resolve_one(base_doc, ext_path)
        for key, value in ext_doc.items():
            if key not in result:
                result[key] = value
    return result
