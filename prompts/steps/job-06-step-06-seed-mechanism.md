# job-06/step-06: Job-06 gate - seed-mechanism support for c3h4 parity

## Current status
Gate runs for superminimal (parity divergent but documented) and c3h4 is BLOCKED-STRUCTURAL because rmgpu does not load seed mechanisms. RMG-Py c3h4 uses `database.seedMechanisms = ['GRI-Mech3.0-N']`, which seeds the core from the GRI library. rmgpu currently only seeds from `species:` block.

## Goal
Make c3h4 parity checkable by implementing rmgdb-backed seed-mechanism loading, then re-run gates/gate_06.py and report results. The gate must run a real `rmgpu run` on `examples/c3h4.yaml`, produce a full PLAN 12.3 output tree, and compare core/edge sets vs the RMG-Py baseline. c3h4 should run end-to-end like superminimal, not remain BLOCKED-STRUCTURAL.

## Changes needed

### 1. Ensure rmgdb seed-mechanism loader exists and works
File: `rmgpu/db/seed_loader.py`
- Resolve mechanism name to kinetics library id in `/home/jackson/rmgpu/rmgdb/db/kinetics.db`
  * `kinetics_libraries_table` → id by name
  * Species: `kinetics_library_dictionary_table` with `library_id`, `label`, `adjacency_list`
  * Reactions: `kinetics_library_reactions_table` + `kinetics_library_reaction_species_table` (species_label + role) + `kinetics_arrhenius_table` for Arrhenius params
- Thermo: `thermo.db` → `thermo_libraries_table` + `thermo_data_table` via `library_parent_id`
- Build `Species` objects:
  * Molecule via `Molecule.from_adjacency_list(adjacency_list)`
  * Label from DB
  * Thermo from DB if available, else leave None for ML estimation
- Build `Reaction` objects:
  * Reactants/products from species_label + role
  * `RateRegistry` with Arrhenius params converted from CGS to SI (`A` *1e-6 for bimolecular)
  * `degeneracy` from DB
- Return `(species_list, reaction_list)`

### 2. Wire loader into main
File: `rmgpu/main.py`
- Add `_build_seed_mechanisms(model_input)`:
  * Read `db_block.seed_mechanisms` from Input
  * For each name, call loader → species + reactions
  * Deduplicate species by canonical key, merge reactions
- Extend `RunContext` to carry `seed_mechanisms_species` and `seed_mechanisms_reactions`
- Update `_build_databases` unchanged; loader uses existing `Databases` paths

### 3. Extend RunContext and CoreEdgeLoop init
File: `rmgpu/core/loop.py`
- Add fields to `RunContext`:
  * `seed_mechanisms_species: List[Species] = []`
  * `seed_mechanisms_reactions: List[Reaction] = []`
- In `CoreEdgeLoop.run()` before seeding from `ctx.seed_species`:
  * Load seed mechanism species/reactions into `self.model.core`
  * Populate `self.species_by_key` for all seed species
  * Add seed reactions to `self.model.core.reactions` and `self.reactions` dict
  * Thermo-estimate seed species if thermo missing (existing `_thermo` path)
- Log: `seed mechanisms: X species / Y reactions`

### 4. Update gates/gate_06.py to run real c3h4
File: `gates/gate_06.py`
- Change `check_c3h4` to call `M.run` on `examples/c3h4.yaml` with output root `examples/run_output_c3h4`
- Parity vs `gates/baselines/c3h4/summary.json` using same canonical key logic as superminimal
- Report core/edge counts, set diffs, and divergence cause
- Remove BLOCKED-STRUCTURAL fast-path; real run must complete

## Checks targeting
- `python gates/gate_06.py` → c3h4 status changes from BLOCKED-STRUCTURAL to PASS/ documented divergence
- `pytest tests/ -q` → all pass
- c3h4 parity vs RMG-Py baseline recorded in `reports/gate_06_results.json`
- Output tree for c3h4 exists per PLAN 12.3
- Output tree for superminimal remains valid

## Notes for next session
- Seed mechanism loading uses rmgdb SQLite only, no RMG-database Python files
- GRI-Mech3 kinetics library id = 30 in rmgdb/db/kinetics.db
- Adjacency list → Molecule conversion uses existing `Molecule.from_adjacency_list`
- Rate params need CGS→SI conversion for A
- Keep existing superminimal parity documented divergence; focus is to unblock c3h4
