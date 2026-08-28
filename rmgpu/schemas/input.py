"""Input schema blocks for rmgpu YAML input documents.

The schema is the de-facto API of the package (PLAN.md 12.4): versioned via the
`rmgpu: 1.0` key, shared between the CLI, the legacy importer (job-03/step-05),
and later jobs. Blocks are LENIENT (extra='allow') so the legacy importer can
preserve every DSL value losslessly; the canonical (strict) usage is the PLAN
12.2 example document.
"""

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
    "LiquidSurfaceReactor",
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
    "GeneratedSpeciesConstraintsBlock",
    "CatalystPropertiesBlock",
    "QuantumMechanicsBlock",
    "Input",
    "resolve_extends",
    "load_input",
]

_VERSION_PATTERN = re.compile(r"^\d+\.\d+$")

# Lenient config: the legacy DSL has a long tail of block kwargs that the
# importer must preserve without loss. Unknown keys are kept (visible in
# model_dump and in `rmgpu run` output), never silently dropped.
_LENIENT = ConfigDict(extra="allow", arbitrary_types_allowed=True)


def _coerce_quantity(v: Any) -> Quantity:
    """Convert a dict, string, tuple, or Quantity to a Quantity."""
    if isinstance(v, Quantity):
        return v
    if isinstance(v, str):
        return Quantity.from_string(v)
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return Quantity(v[0], v[1])
    if isinstance(v, dict):
        return Quantity(v["value"], v["unit"])
    raise ValueError(f"Invalid quantity input: {v!r}")


class StructureValue(pydantic.BaseModel):
    """A structure specification: SMILES/InChI/adjlist string, or a dict naming one."""

    smiles: Optional[str] = None
    inchi: Optional[str] = None
    adjlist: Optional[str] = None
    group_adjlist: Optional[str] = None
    fragment_adjlist: Optional[str] = None
    smarts: Optional[str] = None
    fragment: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "StructureValue":
        if (self.smiles is None and self.inchi is None and self.adjlist is None
                and self.group_adjlist is None and self.fragment_adjlist is None
                and self.smarts is None):
            raise ValueError("structure must provide smiles, inchi, adjlist, group_adjlist, fragment_adjlist, or smarts")
        return self

    @property
    def value(self) -> str:
        return self.smiles or self.inchi or self.adjlist or self.group_adjlist or self.fragment_adjlist or self.smarts


class DatabaseBlock(pydantic.BaseModel):
    """Database configuration block.

    Library lists accept plain names, or legacy ``(name, include_reverse)``
    pairs preserved by the importer.
    """

    model_config = _LENIENT

    thermo_libraries: Optional[Union[List[Any], str]] = None
    reaction_libraries: Optional[Union[List[Any], str]] = None
    seed_mechanisms: Optional[Union[List[Any], str]] = None
    kinetics_families: Union[Literal["default"], List[str], str] = "default"
    kinetics_depositories: Optional[Union[Literal["default", "all"], List[Any], str]] = None
    kinetics_estimator: Literal["ml", "library", "rate rules"] = "ml"
    transport_libraries: Union[Literal["auto"], List[Any], str] = "auto"


class Species(pydantic.BaseModel):
    """A chemical species declaration."""

    model_config = _LENIENT

    label: str = Field(..., pattern=r"^[^+]+$")
    reactive: bool = True
    structure: Union[str, StructureValue]
    thermo: Optional[Dict[str, Any]] = None
    kinetics: Optional[Dict[str, Any]] = None
    constraints: Optional[List[str]] = None

    @field_validator("structure", mode="after")
    @classmethod
    def _coerce_structure(cls, v: Any) -> StructureValue:
        if isinstance(v, str):
            return StructureValue(smiles=v)
        return v


class ForbiddenEntry(pydantic.BaseModel):
    """A forbidden structure entry."""

    model_config = _LENIENT

    label: Optional[str] = None
    structure: Union[str, StructureValue]
    reason: Optional[str] = None

    @field_validator("structure", mode="after")
    @classmethod
    def _coerce_structure(cls, v: Any) -> StructureValue:
        # Mirror Species: accept the legacy DSL / canonical bare SMILES
        # string (and a dict naming one of the structure fields) so
        # forbidden() entries round-trip losslessly. The field's JSON schema
        # keeps both string and object forms via the Union.
        if isinstance(v, str):
            return StructureValue(smiles=v)
        if isinstance(v, dict):
            return StructureValue(**v)
        return v


class TerminationCondition(pydantic.BaseModel):
    """Termination conditions for a reactor.

    `conversion` accepts the legacy DSL form {species: fraction} as well as the
    canonical {species, value} map.
    """

    model_config = _LENIENT

    conversion: Optional[Dict[str, Any]] = None
    time: Optional[Quantity] = None
    rate_ratio: Optional[float] = None
    criticality: Optional[float] = None

    @field_validator("time", mode="before")
    @classmethod
    def _convert_time(cls, v: Any) -> Quantity:
        return _coerce_quantity(v)

    @model_validator(mode="after")
    def _validate(self) -> "TerminationCondition":
        if self.conversion is None and self.time is None and self.rate_ratio is None \
                and self.criticality is None:
            raise ValueError("At least one termination condition is required")
        return self


class SimpleReactor(pydantic.BaseModel):
    """Simple reactor with fixed T and P."""

    model_config = _LENIENT

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

    model_config = _LENIENT

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

    model_config = _LENIENT

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

    model_config = _LENIENT

    type: Literal["liquid"] = "liquid"
    temperature: Quantity
    initial_concentrations: Dict[str, Any]
    termination: Optional[TerminationCondition] = None
    n_sims: Optional[int] = None
    sensitivity: Optional[List[str]] = None
    sensitivity_threshold: Optional[float] = None
    constant_species: Optional[List[str]] = None
    liquid_volume: Optional[Quantity] = None
    residence_time: Optional[Quantity] = None
    inlet_volumetric_flow_rate: Optional[Quantity] = None
    outlet_volumetric_flow_rate: Optional[Quantity] = None
    inlet_concentrations: Optional[Dict[str, Any]] = None
    vapor_pressure: Optional[Quantity] = None
    vapor_mole_fractions: Optional[Dict[str, float]] = None

    @field_validator("temperature", "liquid_volume", "residence_time",
                     "inlet_volumetric_flow_rate", "outlet_volumetric_flow_rate",
                     "vapor_pressure", mode="before")
    @classmethod
    def _convert_quantity(cls, v: Any) -> Quantity:
        return _coerce_quantity(v)


class MBSampledReactor(pydantic.BaseModel):
    """Monte Carlo sampled reactor."""

    model_config = _LENIENT

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

    model_config = _LENIENT

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


class LiquidSurfaceReactor(pydantic.BaseModel):
    """Liquid-surface (electrochemical) reactor: liquid phase + surface sites."""

    model_config = _LENIENT

    type: Literal["liquid_surface"] = "liquid_surface"
    temperature: Quantity
    initial_concentrations: Dict[str, Any]
    initial_surface_coverages: Dict[str, float]
    surface_volume_ratio: Quantity
    liq_potential: Optional[Quantity] = None
    surf_potential: Optional[Quantity] = None
    distance: Optional[Quantity] = None
    viscosity: Optional[Quantity] = None
    constant_species: Optional[List[str]] = None
    termination: Optional[TerminationCondition] = None

    @field_validator("temperature", "surface_volume_ratio", "liq_potential",
                     "surf_potential", "distance", "viscosity", mode="before")
    @classmethod
    def _convert_quantity(cls, v: Any) -> Quantity:
        return _coerce_quantity(v)


class StagedReactor(SimpleReactor):
    """Staged reactor: per-stage temperature/pressure and/or composition.

    Covers the legacy DSL's list-valued temperature/pressure in a
    simpleReactor() call (e.g. SR_test, halogens/2-BTP, regression/oxidation).
    """

    model_config = _LENIENT

    type: Literal["simple", "staged"] = "staged"
    temperature: Optional[Quantity] = None
    pressure: Optional[Quantity] = None
    initial_mole_fractions: Optional[Dict[str, Any]] = None
    staged_temperatures: Optional[List[Quantity]] = None
    staged_pressures: Optional[List[Quantity]] = None
    staged_initial_mole_fractions: Optional[List[Dict[str, Any]]] = None

    @field_validator("staged_temperatures", "staged_pressures", mode="before")
    @classmethod
    def _convert_staged_quantities(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [_coerce_quantity(x) for x in v]
        return v

    @model_validator(mode="after")
    def _validate_stages(self) -> "StagedReactor":
        if not self.staged_initial_mole_fractions and not self.staged_temperatures \
                and not self.staged_pressures:
            raise ValueError("staged reactor requires staged_initial_mole_fractions, staged_temperatures, or staged_pressures")
        return self


class LiquidStagedReactor(LiquidReactor):
    """Staged liquid reactor with per-stage temperature."""

    model_config = _LENIENT

    type: Literal["liquid", "staged"] = "staged"
    staged_temperatures: Optional[List[Quantity]] = None

    @field_validator("staged_temperatures", mode="before")
    @classmethod
    def _convert_staged_temperatures(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [_coerce_quantity(x) for x in v]
        return v


class ConstantVStagedReactor(ConstantVReactor):
    """Staged constant-V reactor with per-stage temperature."""

    model_config = _LENIENT

    type: Literal["const_V", "staged"] = "staged"
    staged_temperatures: Optional[List[Quantity]] = None

    @field_validator("staged_temperatures", mode="before")
    @classmethod
    def _convert_staged_temperatures(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [_coerce_quantity(x) for x in v]
        return v


class PressureStagedReactor(ConstantTPReactor):
    """Staged reactor with per-stage temperature and pressure."""

    model_config = _LENIENT

    type: Literal["staged"] = "staged"
    staged_temperatures: Optional[List[Quantity]] = None
    staged_pressures: Optional[List[Quantity]] = None

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
    LiquidSurfaceReactor,
    StagedReactor,
    LiquidStagedReactor,
    ConstantVStagedReactor,
    PressureStagedReactor,
]]

class SimulatorBlock(pydantic.BaseModel):
    """Simulator tolerances."""

    model_config = _LENIENT

    atol: float = 1e-16
    rtol: float = 1e-8


class ModelBlock(pydantic.BaseModel):
    """Core/edge model tolerance settings."""

    model_config = _LENIENT

    tolerance_move_to_core: float = 0.1
    tolerance_keep_in_edge: float = 0.0
    tolerance_interrupt_simulation: float = 1.0
    maximum_edge_species: int = 100000
    filter_reactions: bool = False


class PressureDependenceBlock(pydantic.BaseModel):
    """Pressure dependence settings."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, populate_by_name=True)

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
    """ML estimator checkpoint references (or a legacy DSL flag)."""

    model_config = _LENIENT

    thermo: Any
    kinetics: Optional[Any] = None


class SolvationBlock(pydantic.BaseModel):
    """Solvation plugin settings."""

    model_config = _LENIENT

    solvent: str
    model: Optional[str] = "smd"


class UncertaintyBlock(pydantic.BaseModel):
    """Uncertainty quantification settings."""

    model_config = _LENIENT

    enabled: bool = False
    species: Optional[List[str]] = None


class OptionsBlock(pydantic.BaseModel):
    """Output options."""

    model_config = _LENIENT

    save_profiles: bool = False
    save_plots: bool = False
    save_edge: bool = False
    units: Literal["si", "cgs"] = "si"


class GeneratedSpeciesConstraintsBlock(pydantic.BaseModel):
    """Constraints on species the model may generate (legacy DSL long tail)."""

    model_config = _LENIENT

    allowed: Optional[List[str]] = None
    maximum_radical_electrons: Optional[int] = None
    maximum_surface_sites: Optional[int] = None
    maximum_carbon_atoms: Optional[int] = None
    maximum_oxygen_atoms: Optional[int] = None
    maximum_nitrogen_atoms: Optional[int] = None
    maximum_silicon_atoms: Optional[int] = None
    maximum_sulfur_atoms: Optional[int] = None
    maximum_heavy_atoms: Optional[int] = None
    maximum_surface_bond_order: Optional[int] = None
    maximum_singlet_carbenes: Optional[int] = None
    maximum_carbene_radicals: Optional[int] = None
    allow_singlet_o2: Optional[bool] = None


class CatalystPropertiesBlock(pydantic.BaseModel):
    """Catalyst properties (metal, binding energies, site density)."""

    model_config = _LENIENT

    metal: Optional[str] = None
    binding_energies: Optional[Dict[str, Any]] = None
    surface_site_density: Optional[Any] = None
    coverage_dependence: Optional[bool] = None


class QuantumMechanicsBlock(pydantic.BaseModel):
    """QM settings, PRESERVED VERBATIM from the legacy DSL.

    QM is out of scope for rmgpu (PLAN 8a.3): this block exists so the legacy
    importer is lossless; rmgpu run-time code must ignore it.
    """

    model_config = _LENIENT

    software: Optional[str] = None
    method: Optional[str] = None
    scratch_directory: Optional[str] = None
    only_cyclics: Optional[bool] = None
    max_radical_number: Optional[int] = None
    file_store: Optional[str] = None


class Input(pydantic.BaseModel):
    """Top-level input document."""

    model_config = _LENIENT

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
    generated_species_constraints: Optional[GeneratedSpeciesConstraintsBlock] = None
    catalyst_properties: Optional[CatalystPropertiesBlock] = None
    quantum_mechanics: Optional[QuantumMechanicsBlock] = None

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
