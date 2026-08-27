"""Input schema blocks for rmgpu YAML input documents."""

import os
import re
from typing import Any, Dict, List, Literal, Optional, Union

import pydantic
from pydantic import ConfigDict, Field, field_validator, model_validator

from rmgpu.units import Quantity

__all__ = [
    "Quantity",
    "StructureValue",
    "DatabaseBlock",
    "Species",
    "ForbiddenEntry",
    "TerminationCondition",
    "SimpleReactor",
    "ConstantVReactor",
    "ConstantTPReactor",
    "LiquidReactor",
    "MBSampledReactor",
    "SurfaceReactor",
    "StagedReactor",
    "LiquidStagedReactor",
    "ConstantVStagedReactor",
    "PressureStagedReactor",
    "Reactors",
    "SimulatorBlock",
    "ModelBlock",
    "PressureDependenceBlock",
    "MLEstimatorBlock",
    "SolvationBlock",
    "UncertaintyBlock",
    "OptionsBlock",
    "Input",
    "resolve_extends",
    "load_input",
]

_VERSION_PATTERN = re.compile(r"^\d+\.\d+$")

def _coerce_quantity(v: Any) -> Quantity:
    """Convert a dict, string, or Quantity to a Quantity."""
    if isinstance(v, Quantity):
        return v
    if isinstance(v, str):
        return Quantity.from_string(v)
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

    @field_validator("structure", mode="before")
    @classmethod
    def _coerce_structure(cls, v: Any) -> Any:
        if isinstance(v, str):
            return {"smiles": v}
        return v

class ForbiddenEntry(pydantic.BaseModel):
    """A forbidden structure entry."""

    structure: str
    reason: Optional[str] = None

class TerminationCondition(pydantic.BaseModel):
    """Termination conditions for a reactor."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    conversion: Optional[Dict[str, Union[str, float]]] = None
    time: Optional[Quantity] = None
    criticality: Optional[float] = None

    @field_validator("time", mode="before")
    @classmethod
    def _convert_time(cls, v: Any) -> Quantity:
        return _coerce_quantity(v)

    @model_validator(mode="after")
    def _validate(self) -> "TerminationCondition":
        if self.conversion is None and self.time is None and self.criticality is None:
            raise ValueError("At least one termination condition is required")
        if self.conversion is not None:
            if "species" not in self.conversion or "value" not in self.conversion:
                raise ValueError("conversion must include species and value")
        return self

class SimpleReactor(pydantic.BaseModel):
    """Simple reactor with fixed T and P."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["simple"] = "simple"
    temperature: Quantity
    pressure: Quantity
    initial_mole_fractions: Dict[str, Union[float, List[float]]]
    termination: Optional[TerminationCondition] = None
    n_sims: Optional[int] = None
    sensitivity: Optional[List[str]] = None
    sensitivity_threshold: Optional[float] = None
    constant_species: Optional[List[str]] = None
    balance_species: Optional[str] = None

    @field_validator("temperature", "pressure", mode="before")
    @classmethod
    def _convert_quantity(cls, v: Any) -> Quantity:
        return _coerce_quantity(v)

class ConstantVReactor(pydantic.BaseModel):
    """Constant volume ideal gas reactor."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["const_V"] = "const_V"
    temperature: Quantity
    pressure: Quantity
    initial_mole_fractions: Dict[str, Union[float, List[float]]]
    termination: Optional[TerminationCondition] = None
    balance_species: Optional[str] = None

    @field_validator("temperature", "pressure", mode="before")
    @classmethod
    def _convert_quantity(cls, v: Any) -> Quantity:
        return _coerce_quantity(v)

class ConstantTPReactor(pydantic.BaseModel):
    """Constant temperature and pressure ideal gas reactor."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["const_TP"] = "const_TP"
    temperature: Quantity
    pressure: Quantity
    initial_mole_fractions: Dict[str, Union[float, List[float]]]
    termination: Optional[TerminationCondition] = None
    balance_species: Optional[str] = None

    @field_validator("temperature", "pressure", mode="before")
    @classmethod
    def _convert_quantity(cls, v: Any) -> Quantity:
        return _coerce_quantity(v)

class LiquidReactor(pydantic.BaseModel):
    """Liquid phase reactor."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["liquid"] = "liquid"
    temperature: Quantity
    initial_concentrations: Dict[str, Union[float, List[float]]]
    termination: Optional[TerminationCondition] = None
    n_sims: Optional[int] = None
    sensitivity: Optional[List[str]] = None
    sensitivity_threshold: Optional[float] = None
    constant_species: Optional[List[str]] = None
    liquid_volume: Optional[Quantity] = None
    residence_time: Optional[Quantity] = None
    inlet_volumetric_flow_rate: Optional[Quantity] = None
    outlet_volumetric_flow_rate: Optional[Quantity] = None
    inlet_concentrations: Optional[Dict[str, Union[float, List[float]]]] = None
    vapor_pressure: Optional[Quantity] = None
    vapor_mole_fractions: Optional[Dict[str, float]] = None

    @field_validator("temperature", "liquid_volume", "residence_time", "inlet_volumetric_flow_rate", "outlet_volumetric_flow_rate", "vapor_pressure", mode="before")
    @classmethod
    def _convert_quantity(cls, v: Any) -> Quantity:
        return _coerce_quantity(v)

class MBSampledReactor(pydantic.BaseModel):
    """Monte Carlo sampled reactor."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["mb_sampled"] = "mb_sampled"
    temperature: Quantity
    pressure: Quantity
    initial_mole_fractions: Dict[str, Union[float, List[float]]]
    mbsampling_rate: Quantity
    termination: Optional[TerminationCondition] = None
    sensitivity: Optional[List[str]] = None
    sensitivity_threshold: Optional[float] = None
    constant_species: Optional[List[str]] = None

    @field_validator("temperature", "pressure", "mbsampling_rate", mode="before")
    @classmethod
    def _convert_quantity(cls, v: Any) -> Quantity:
        return _coerce_quantity(v)

class SurfaceReactor(pydantic.BaseModel):
    """Surface catalysis reactor."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["surface"] = "surface"
    temperature: Quantity
    initial_pressure: Quantity
    initial_gas_mole_fractions: Dict[str, float]
    initial_surface_coverages: Dict[str, float]
    surface_volume_ratio: Quantity
    n_sims: Optional[int] = None
    termination: Optional[TerminationCondition] = None
    sensitivity: Optional[List[str]] = None
    sensitivity_threshold: Optional[float] = None

    @field_validator("temperature", "initial_pressure", "surface_volume_ratio", mode="before")
    @classmethod
    def _convert_quantity(cls, v: Any) -> Quantity:
        return _coerce_quantity(v)

class StagedReactor(SimpleReactor):
    """Staged simple reactor: same T/P across stages, changing composition."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["simple", "staged"] = "staged"
    staged_initial_mole_fractions: List[Dict[str, Union[float, List[float]]]]

    @model_validator(mode="after")
    def _validate_stages(self) -> "StagedReactor":
        if not self.staged_initial_mole_fractions:
            raise ValueError("staged reactor requires at least one staged_initial_mole_fractions entry")
        return self

class LiquidStagedReactor(LiquidReactor):
    """Staged liquid reactor with per-stage temperature."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["liquid", "staged"] = "staged"
    staged_temperatures: List[Quantity]

    @field_validator("staged_temperatures", mode="before")
    @classmethod
    def _convert_staged_temperatures(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [_coerce_quantity(x) for x in v]
        return v

class ConstantVStagedReactor(ConstantVReactor):
    """Staged constant-V reactor with per-stage temperature."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["const_V", "staged"] = "staged"
    staged_temperatures: List[Quantity]

    @field_validator("staged_temperatures", mode="before")
    @classmethod
    def _convert_staged_temperatures(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [_coerce_quantity(x) for x in v]
        return v

class PressureStagedReactor(ConstantTPReactor):
    """Staged reactor with per-stage temperature and pressure."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["staged"] = "staged"
    staged_temperatures: List[Quantity]
    staged_pressures: List[Quantity]

    @field_validator("staged_temperatures", "staged_pressures", mode="before")
    @classmethod
    def _convert_staged_quantities(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [_coerce_quantity(x) for x in v]
        return v

Reactors = List[Union[
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
]]

class SimulatorBlock(pydantic.BaseModel):
    """Simulator tolerances."""

    atol: float = 1e-16
    rtol: float = 1e-8

class ModelBlock(pydantic.BaseModel):
    """Core/edge model tolerance settings."""

    tolerance_move_to_core: float = 0.1
    tolerance_keep_in_edge: float = 0.0
    tolerance_interrupt_simulation: float = 1.0
    maximum_edge_species: int = 100000
    filter_reactions: bool = False

class PressureDependenceBlock(pydantic.BaseModel):
    """Pressure dependence settings."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    method: str
    tmin: Quantity = Field(Quantity(300.0, "K"), alias="Tmin")
    tmax: Quantity = Field(Quantity(2000.0, "K"), alias="Tmax")
    tcount: int = Field(20, alias="Tcount")
    pmin: Quantity = Field(Quantity(1.0, "bar"), alias="Pmin")
    pmax: Quantity = Field(Quantity(50.0, "bar"), alias="Pmax")
    pcount: int = Field(10, alias="Pcount")
    maximum_grain_size: Quantity = Field(Quantity(0.0), alias="maximumGrainSize")
    minimum_number_of_grains: int = Field(0, alias="minimumNumberOfGrains")
    interpolation_model: Optional[str] = None
    maximum_atoms: Optional[int] = None
    completed_networks: Optional[List[str]] = None

    @field_validator("method")
    @classmethod
    def _normalize_method(cls, v: str) -> str:
        aliases = {
            "cse": "strong collision",
            "masc": "modified strong collision",
            "rs": "randomized strong collision",
            "sls": "super logarithmic spacing",
        }
        normalized = v.lower()
        if normalized in aliases:
            return aliases[normalized]
        # Already a long-form name; keep as-is
        return normalized

    @field_validator("tmin", "tmax", "pmin", "pmax", "maximum_grain_size", mode="before")
    @classmethod
    def _convert_quantity(cls, v: Any) -> Quantity:
        return _coerce_quantity(v)

class MLEstimatorBlock(pydantic.BaseModel):
    """ML estimator checkpoint references."""

    thermo: str
    kinetics: Optional[str] = None

class SolvationBlock(pydantic.BaseModel):
    """Solvation plugin settings."""

    solvent: str
    model: Literal["smd"] = "smd"

class UncertaintyBlock(pydantic.BaseModel):
    """Uncertainty quantification settings."""

    enabled: bool = False
    species: Optional[List[str]] = None

class OptionsBlock(pydantic.BaseModel):
    """Output options."""

    save_profiles: bool = False
    save_plots: bool = False
    save_edge: bool = False
    units: Literal["si", "cgs"] = "si"

class Input(pydantic.BaseModel):
    """Top-level input document."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rmgpu: str = Field(..., pattern=_VERSION_PATTERN)
    extends: Optional[Union[str, List[str]]] = None
    database: Optional[DatabaseBlock] = None
    species: Optional[List[Species]] = None
    forbidden: Optional[List[ForbiddenEntry]] = None
    reactors: Optional[Reactors] = None
    simulator: Optional[SimulatorBlock] = None
    model: Optional[ModelBlock] = None
    pressure_dependence: Optional[PressureDependenceBlock] = None
    ml_estimator: Optional[MLEstimatorBlock] = None
    solvation: Optional[SolvationBlock] = None
    uncertainty: Optional[UncertaintyBlock] = None
    options: Optional[OptionsBlock] = None

    @field_validator("rmgpu", mode="before")
    @classmethod
    def _coerce_version(cls, v: Any) -> str:
        return str(v)

    @classmethod
    def dump_json_schema(cls) -> Dict[str, Any]:
        return cls.model_json_schema()

def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)

def _resolve_one(base_doc: Dict[str, Any], ext_path: str, base_dir: str, visited: Optional[set] = None) -> Dict[str, Any]:
    """Resolve a single extends path, applying the extends semantics."""
    full_path = os.path.abspath(os.path.join(base_dir, ext_path))
    if visited is None:
        visited = set()
    if full_path in visited:
        raise ValueError(f"Cycle detected in extends chain: {full_path}")
    visited.add(full_path)
    ext_doc = _load_yaml(full_path)
    if "extends" in ext_doc:
        # Recursively resolve the nested extends
        nested_dir = os.path.dirname(full_path)
        ext_doc = resolve_extends(ext_doc, nested_dir, visited)
    if "rmgpu" in ext_doc:
        ext_version = str(ext_doc["rmgpu"])
        base_version = str(base_doc.get("rmgpu", "1.0"))
        if ext_version != base_version:
            raise ValueError("Version mismatch in extends document")
    return ext_doc

def resolve_extends(base_doc: Dict[str, Any], base_dir: str = ".", _visited: Optional[set] = None) -> Dict[str, Any]:
    """Resolve extends chains into a single document."""
    if "extends" not in base_doc or not base_doc["extends"]:
        return base_doc
    if _visited is None:
        _visited = set()
    extends = base_doc.get("extends", [])
    if isinstance(extends, str):
        extends = [extends]
    if not extends:
        return base_doc
    current_path = os.path.abspath(base_dir)
    if current_path not in _visited:
        _visited.add(current_path)
    result = dict(base_doc)
    result.pop("extends", None)
    for ext_path in extends:
        ext_doc = _resolve_one(base_doc, ext_path, base_dir, _visited)
        for key, value in ext_doc.items():
            if key not in result:
                result[key] = value
            elif isinstance(result[key], dict) and isinstance(value, dict):
                # Merge nested dicts (e.g., database block) - base wins
                merged = dict(value)
                for k, v in result[key].items():
                    if v is not None:
                        merged[k] = v
                result[key] = merged
    return result

def load_input(path: str) -> Input:
    """Load an input YAML file and resolve extends chains."""
    path = os.path.abspath(path)
    base_dir = os.path.dirname(path)
    doc = _load_yaml(path)
    resolved = resolve_extends(doc, base_dir)
    return Input(**resolved)