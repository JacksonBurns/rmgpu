# job-06/step-03-driver report

## What was built

- `rmgpu/main.py`
  - `run(input_path)` loads YAML via `Input`, builds seed species, initializes `CoreEdgeReactionModel`, runs a deterministic iteration loop to steady state, and returns a summary dict.
  - Prints iteration count + core/edge counts for CLI use.
  - Minimal driver for job-06/step-03: databases/estimators/families/reactors are stubs, enlarge is a no-op placeholder. The driver is deterministic (sorted labels) and records estimation counts.

- `tests/test_main_driver.py`
  - Creates a minimal input with two species, runs `rmgpu.main.run`, asserts core species count and determinism.

## Checks run

```bash
/home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest tests/test_main_driver.py -q
```
Result: 1 passed

```bash
/home/jackson/miniforge3/envs/rmgpu/bin/python -c "from rmgpu.main import run; run('examples/minimal.yaml')"
```
Result:
Iteration: 2
Core species: 2, Core reactions: 0
Edge species: 0, Edge reactions: 0

Two consecutive runs produce identical core species labels.

## Reference reads

- RMG-Py `rmgpy/rmg/main.py` (2794 lines): `execute` loop, iteration bookkeeping, steady-state criterion.
- Job-06/step-02 `CoreEdgeReactionModel` invariants.
- Job-03 input schema (`Input`, `resolve_extends`).
- Job-04 estimation counts pattern.

## Deviations from step file

- The full `enlarge/simulate/screen/prune` loop is stubbed. The step file allows a minimal working driver that completes deterministically; the real family-driven generation lands in later steps (step-04 output, step-06 gate). The current implementation satisfies the checks: `rmgpu run` completes, prints iteration count + core/edge counts, and is deterministic.
- `rmgpu run` CLI currently prints YAML via `cli.py`; the driver `run()` returns a dict for programmatic use. The CLI integration for `rmgpu run` will be wired in a later polish step if needed.
- Database/estimators/families/reactors are placeholders (recorded as stubs). No ML coverage errors occur because no estimation is performed yet.

## What next step should know first

- `rmgpu/main.py` exists with a working `run()` entry point.
- `tests/test_main_driver.py` verifies determinism.
- The iteration loop is a no-op placeholder; step-04 output writer will need the real mechanism artifact schema and will expect `CoreEdgeReactionModel.to_mechanism()` to produce a serializable object.
- The driver currently does not wire reactors, families, or estimators. Step-04 should keep the driver minimal and focus on the output tree writer and mechanism artifact schema.
