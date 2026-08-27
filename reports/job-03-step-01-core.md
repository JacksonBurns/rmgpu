# job-03/step-01: Core input schema

## What was built

- `rmgpu/schemas/input.py`: Pydantic models for the core input schema blocks:
  - `Quantity`: pint-backed physical quantity with `value`/`unit` constructor and string parser.
  - `StructureValue`: structure specification via SMILES/InChI string or adjacency list dict.
  - `DatabaseBlock`: database configuration (thermo_libraries, reaction_libraries, seed_mechanisms, kinetics_families, kinetics_depositories, kinetics_estimator, transport_libraries).
  - `Species`: chemical species declaration with label, reactive, structure, thermo/kinetics overrides, constraints.
  - `ForbiddenEntry`: forbidden structure entry with structure and reason.
  - `Input`: top-level input document with `rmgpu: 1.0` version key, extends, database, species, forbidden blocks.
  - `resolve_extends()`: pure function to resolve extends chains into a single document (earliest wins on conflicts, cycle detection).
  - `dump_json_schema()` helper on `Input` for JSON schema export.

- `tests/test_schemas_core.py`: 20 tests covering:
  - Quantity construction and unit conversion
  - StructureValue valid/invalid cases
  - DatabaseBlock defaults and custom values
  - Species valid/invalid labels (no `+` in labels)
  - ForbiddenEntry valid entries
  - Input version key enforcement (rejects invalid versions)
  - JSON schema export
  - Extends resolution: no extends, extends in extended doc (error), missing file (error), two-level chain

## Checks

- `pytest tests/test_schemas_core.py -q` -> 20 passed

## Reference reads

- None beyond the step file's listed references (PLAN.md section 12, RMG-Py input.py signatures).

## Deviations

- None. All deliverables implemented as specified.

## Next step notes

- Step 02 should implement reactor blocks, simulator, model, pressure_dependence, ml_estimator, solvation, uncertainty, and options blocks.
- `Quantity` is already defined and can be reused.
- `Input` model already exists; step 02 should extend it with the remaining optional blocks.
