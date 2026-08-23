# job-03/step-01: Input schema: core blocks (quantity, database, species, forbidden)

Job: job-03 - YAML input schema + CLI + legacy .py importer
Prereq: job-01 done (Molecule for structure parsing). job-02 not strictly required, but the database: block references library names from it.
This step is part of that job. The job's overall goal:
Replace RMG's "execute a Python file to configure the run" with a declarative, schema-validated YAML document + a small CLI + a lossless importer for the legacy .py DSL. This is the user-facing front door. Deliverables: rmgpu/schemas/input.py (pydantic), rmgpu/cli.py (click), rmgpu/importer/legacy.py (ast-based).
The job's gate (run by the job's final step):
gates/gate_03.py: 1. DSL inventory: 47 legacy input.py files x functions used (report table). 2. Import: N of 47 files import to schema-valid YAML with zero dropped values; the rest have documented IMPORT-NOTEs. Target: N == 47. 3. `rmgpu validate` on all 47 imported YAMLs: all pass. 4. `rmgpu run minimal.yaml` (the imported minimal example) prints the resolved document. 5. JSON schema exports and validates a hand-written minimal input.yaml (the example from PLAN.md 12.2).

First step of this job: skim /home/jackson/rmgpu/rmgpu/ORIENTATION.md once
(what stays/goes/external; the master-equation data path; conventions). Later
steps of this job do not need it - everything they need is in their own file.

## Context (invariant every step)

- You are a FRESH subagent session; you have no other context. This file plus
  what it points at is everything you need.
- Work in /home/jackson/rmgpu/rmgpu (git, branch main). Reference repos (read-
  only, at /home/jackson/rmgpu): RMG-Py, RMG-database, rmgdb, chemprop_example.
- Interpreter: always the env's python directly,
  /home/jackson/miniforge3/envs/rmgpu/bin/python (created in job 00).
  NEVER install into system python or the active venv.
- ONE subagent at a time; this session MUST NOT spawn subagents. Single heavy
  GPU task at a time on this machine; never kill processes you did not start.
- No Cython, no numba, no QM, no Arkane, no fallback estimators, no second
  reactor backend. SI units internally (J, K, Pa, mol) via pint.
- The existing Chemprop-based estimators in RMG-Py (rmgpy/ml/estimator.py)
  are being REPLACED by rmgpu's new ones - they are reference-only (layout,
  cutoffs, DSL wiring). rmgpu's estimators are re-implemented per
  /home/jackson/rmgpu/chemprop_example/predicting.ipynb. See README.md,
  "The ML estimators are NEW".

## Where you are

- Previous step (already done, its code is in the tree): (this is the first step of the job)
- This step: job-03/step-01-core
- Next step (do NOT start it): job-03/step-02-blocks

## Goal

Define the pydantic input schema, part 1: the Quantity type and the
core blocks (database, species, forbidden). PLAN.md section 12 is the design
- read it in full first (it is the spec).

## Reference to read (this step's budget)

  /home/jackson/rmgpu/rmgpu/PLAN.md  section 12.1-12.5 (the I/O
    redesign: principles, the example input.yaml, quantity syntax,
    extends/layering, CLI, output tree)
  RMG-Py/rmgpy/rmg/input.py  (2057 - the 41-function legacy DSL; read the
    signatures + docstrings of database/species/forbidden/SMILES/
    adjacency_list/react and the Species handling - the importer (step 4)
    maps these)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/schemas/input.py: pydantic models, versioned (`rmgpu: 1.0`
  top key):
    Quantity: {value, unit} map OR "1350 K" string (pint-backed; the parser
    reuses rmgpu.units from job 01); SI-internal.
    DatabaseBlock: thermo_libraries, reaction_libraries, seed_mechanisms
    (paths), kinetics_families ('default' or list), kinetics_depositories,
    kinetics_estimator ('ml' - the PoC default - or 'library'),
    transport_libraries ('auto' or list).
    Species: {label, reactive, structure (smiles/inchi string or
    {adjlist: "..."}), optional per-species thermo/kinetics overrides,
    constraints}.
    ForbiddenEntry: {structure, reason}.
    Top-level Input model: rmgpu version key + the blocks (database,
    species, forbidden; the rest land in step 2 as None-allowed).
  - every field gets a docstring; models export a JSON schema (a
    `dump_json_schema()` helper; the CLI subcommand is step 3).
  - `extends`: field on the top model (path or list); resolution is step 3's
    (CLI) but define the resolution SEMANTICS here (earliest wins on
    conflicts, cycle detection) as a pure function with tests.
- tests/test_schemas_core.py: valid/invalid Quantity, species from SMILES/
  adjlist, database block defaults, version-key enforcement, extends
  resolution (2-level chain + cycle error).

## Checks (must run and pass before you claim done)

  pytest tests/test_schemas_core.py -q -> all pass

## Pitfalls

- The schema is the API (PLAN.md 12.4): no heavy deps in the import
  path; keep the models importable without torch/rdkit where possible
  (structure is a string here; parsing happens at load, not validate).
- Do NOT exec/eval anything - declarative data only (PLAN.md 12.1.1).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-03/step-01: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-03/step-01 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-03-step-01-core.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
