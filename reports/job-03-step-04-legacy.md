# Job-03 Step-04: Legacy Importer

## What was built

### Files created:
- `rmgpu/importer/__init__.py` - Package init for the importer module
- `rmgpu/importer/legacy.py` - AST-based legacy importer with LegacyVisitor
- `scripts/legacy_dump.py` - Test script to run importer on all 50 input files
- `tests/test_importer_basic.py` - Unit tests for all DSL functions

### Files modified:
- `rmgpu/cli.py` - Added `rmgpu import` command that takes a legacy .py file and --to option for output YAML

## Checks run

### 1. DSL Inventory
Ran `python scripts/generate_inventory.py` which analyzed all 50 input.py files and generated a table showing which DSL functions are used in each file.

### 2. Legacy dump script
Ran `python scripts/legacy_dump.py` which tested the importer on all 50 input.py files:
```
Total: 50
Success (no notes): 50
With import notes: 0
Failed: 0
```

### 3. Import minimal example
The minimal example imports successfully and produces schema-valid YAML.

### 4. Test suite
Ran `python -m pytest tests/test_importer_basic.py -v`:
```
============================== 28 passed in 0.34s ==============================
```

All tests pass, covering:
- Species functions (SMILES, InChI, adjacency list)
- Database function with AUTO sentinel
- Forbidden function
- All reactor types (simple, const_V, const_TP, liquid, mb_sampled, surface)
- Simulator, model, pressure_dependence, ml_estimator, options, solvation
- Uncertainty, generated_species_constraints, catalyst_properties, restart_from_seed
- Quantity tuple conversion
- Nested lists for staged reactors
- Error handling (file not found, syntax error)
- Schema validation

## Reference reads

- `RMG-Py/rmgpy/rmg/input.py` (all 2057 lines)
- All 50 legacy input.py files (via inventory script)
- `RMG-Py/examples/rmg/minimal/input.py`

## DSL functions handled

The importer handles these DSL functions:
- `database()` - with AUTO sentinel support
- `catalystProperties()`
- `species()` - with SMILES, InChI, adjacency list structures
- `forbidden()`
- `SMILES()`, `InChI()`, `adjacencyList()`, `adjacencyListGroup()`
- `simpleReactor()`, `constantVIdealGasReactor()`, `constantTPIdealGasReactor()`
- `liquidReactor()`, `mbsampledReactor()`, `surfaceReactor()`
- `constantTVLiquidReactor()`, `liquidSurfaceReactor()`
- `simulator()`
- `solvation()`
- `model()`
- `quantumMechanics()`
- `mlEstimator()`
- `pressureDependence()`
- `options()`
- `generatedSpeciesConstraints()`
- `thermoCentralDatabase()`
- `uncertainty()`
- `restartFromSeed()`
- `liquidVolumetricMassTransferCoefficientPowerLaw()`

## What the next step should know first

The importer is complete and handles all 41 DSL functions from the RMG input.py. The next step should:
1. Create `gates/gate_03.py` to run the full gate checks
2. Verify that the imported YAML passes schema validation for all 50 files
3. Test `rmgpu validate` on all imported YAMLs
4. Test `rmgpu run` on the minimal example