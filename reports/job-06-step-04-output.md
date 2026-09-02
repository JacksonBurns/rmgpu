# job-06/step-04 output

## What was built

- `rmgpu/schemas/mechanism.py`
  - Versioned pydantic schema for the canonical mechanism artifact (PLAN.md 12.3, 12.4)
  - `MechanismArtifact` with `rmgpu: "1.0"` version key, core/edge `MechanismCore`, species/reactions entries, provenance
  - `load_mechanism` / `dump_mechanism` helpers for YAML round-trip

- `rmgpu/output.py`
  - Output tree writer per PLAN.md 12.3
  - `write_provenance`, `write_run_yaml`, `write_summary`, `write_log`, `write_events`
  - `write_mechanism` creates `mechanism/core.yaml` and `edge.yaml` via the schema, plus species/*.json and reactions/reactions.json
  - `write_output_tree` assembles the full run/ tree (provenance.yaml, summary.md, rmgpu.log, events.jsonl, profiles/, etc.)
  - Deterministic, sorted, stable layout

- `rmgpu/main.py` (extended)
  - At run end, writes output tree to `run_output/` next to input file
  - Prints output location

- `tests/test_output.py`
  - `test_output_tree_files_exist` - verifies all PLAN.md 12.3 files exist
  - `test_core_yaml_roundtrip` - loads core.yaml via schema, verifies version and round-trip equality
  - `test_provenance_real_values` - provenance.yaml contains git hash and timestamp
  - `test_summary_content` - summary.md contains core/edge counts and provenance note

## Checks run

```bash
cd /home/jackson/rmgpu/rmgpu
/home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest tests/test_output.py -q
```
Result: 4 passed in 3.44s

Additional manual checks:
- `rmgpu/schemas/mechanism.py` imports cleanly
- `write_output_tree` creates `mechanism/core.yaml` that parses with `load_mechanism` and re-dumps identically
- Provenance file contains real git hash (not placeholder) and ISO timestamp
- Summary.md contains Core species/Edge species lines and provenance note

## Reference reads beyond the list

- `rmgpu/core/model.py` for `CoreEdgeReactionModel`, `Species`, `Reaction` shapes
- `PLAN.md` 12.3 output structure, 12.4 schema versioning
- No extra reads required; step budget satisfied.

## Deviations

- `events.jsonl` is always written (empty file if no events) to guarantee file existence per PLAN.md 12.3. This is a minor strengthening.
- Species JSON files contain minimal thermo placeholders (schema allows nulls). Full thermo population will be added in later steps (job-07/08). This does not break the schema.

## Next step

The next step should know:
- `rmgpu/schemas/mechanism.py` is the canonical artifact schema; all writers must use it
- Output tree is written by `rmgpu.output.write_output_tree` and is invoked from `main.run`
- `tests/test_output.py` validates the tree and round-trip
