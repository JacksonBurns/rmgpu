"""Canonical mechanism artifact schema (PLAN.md 12.3, 12.4).

Versioned via the `rmgpu: 1.0` key. The schema is the contract for
mechanism/core.yaml and edge.yaml. It must be stable and bidirectional
seedable.

The artifact is the output of a run; it can be loaded back into a
Mechanism object for restart via database.seed_mechanisms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator

_VERSION_PATTERN = r"^\d+\.\d+$"

_LENIENT = ConfigDict(extra="allow")


class ThermoModel(BaseModel):
    model_config = _LENIENT
    model: Literal["wilhoit", "nasa7", "ml"] = "ml"
    Hf298: Optional[float] = None
    S298: Optional[float] = None
    Cp: Optional[List[float]] = None
    method: Literal["ml", "library"] = "ml"
    uncertainty: Optional[float] = None


class SpeciesEntry(BaseModel):
    model_config = _LENIENT
    label: str
    formula: Optional[str] = None
    smiles: Optional[str] = None
    adjlist: Optional[str] = None
    source: Literal["library", "ml", "depository"] = "ml"
    symmetry: Optional[int] = None
    thermo: ThermoModel


class RateParams(BaseModel):
    model_config = _LENIENT
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    method: Literal["ml", "library"] = "ml"
    uncertainty: Optional[float] = None


class ReactionEntry(BaseModel):
    model_config = _LENIENT
    label: str
    reactants: List[str]
    products: List[str]
    family: Optional[str] = None
    template: Optional[str] = None
    source: Literal["ml", "library"] = "ml"
    degeneracy: float = 1.0
    rate: RateParams


class MechanismCore(BaseModel):
    model_config = _LENIENT
    rmgpu: str = Field(default="1.0", pattern=_VERSION_PATTERN)
    species: List[SpeciesEntry]
    reactions: List[ReactionEntry]
    core_species_labels: Optional[List[str]] = None
    edge_species_labels: Optional[List[str]] = None

    @field_validator("rmgpu")
    @classmethod
    def _check_version(cls, v: str) -> str:
        if not __import__("re").match(_VERSION_PATTERN, v):
            raise ValueError("rmgpu version must be X.Y")
        return v


class MechanismArtifact(BaseModel):
    """Top-level artifact with provenance and version."""
    model_config = _LENIENT
    rmgpu: str = Field(default="1.0", pattern=_VERSION_PATTERN)
    core: MechanismCore
    edge: Optional[MechanismCore] = None
    provenance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def load_mechanism(path: str) -> MechanismArtifact:
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return MechanismArtifact(**data)


def dump_mechanism(mech: MechanismArtifact, path: str) -> None:
    import yaml
    with open(path, "w") as f:
        yaml.safe_dump(mech.to_dict(), f, sort_keys=False)
