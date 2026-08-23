# job-03/step-03: CLI: run/validate/schema/version

Job: job-03 - YAML input schema + CLI + legacy .py importer
Prereq: job-01 done (Molecule for structure parsing). job-02 not strictly required, but the database: block references library names from it.
This step is part of that job. The job's overall goal:
Replace RMG's "execute a Python file to configure the run" with a declarative, schema-validated YAML document + a small CLI + a lossless importer for the legacy .py DSL. This is the user-facing front door. Deliverables: rmgpu/schemas/input.py (pydantic), rmgpu/cli.py (click), rmgpu/importer/legacy.py (ast-based).
The job's gate (run by the job's final step):
gates/gate_03.py: 1. DSL inventory: 47 legacy input.py files x functions used (report table). 2. Import: N of 47 files import to schema-valid YAML with zero dropped values; the rest have documented IMPORT-NOTEs. Target: N == 47. 3. `rmgpu validate` on all 47 imported YAMLs: all pass. 4. `rmgpu run minimal.yaml` (the imported minimal example) prints the resolved document. 5. JSON schema exports and validates a hand-written minimal input.yaml (the example from PLAN.md 12.2).

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

- Previous step (already done, its code is in the tree): job-03/step-02-blocks
- This step: job-03/step-03-cli
- Next step (do NOT start it): job-03/step-04-legacy

## Goal

The click CLI: `rmgpu run` (load + resolve + print the document),
`rmgpu validate` (report ALL problems at once), `rmgpu schema` (JSON schema
export), `rmgpu version`. diff/export/inspect are later jobs (stub with "not
yet implemented").

## Reference to read (this step's budget)

  (the schema from steps 1-2; click; no new RMG-Py reads)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/cli.py (replace the stubs):
    `rmgpu run input.yaml` - load, validate, print the resolved document as
    YAML and exit 0 (actual execution is job 06+).
    `rmgpu validate input.yaml` - validate only, collect ALL problems (no
    fail-fast), print them with locations, exit 0/1.
    `rmgpu schema --out path.json` - export the JSON schema.
    `rmgpu version` - the version.
    Stubs: `rmgpu import` (step 4 lands it), `rmgpu export/diff/inspect`
    ("not yet implemented").
- A hand-written minimal input.yaml at examples/minimal.yaml (the PLAN.md
  12.2 example, trimmed) - the running example for the job.
- tests/test_cli.py: run prints valid YAML; validate on a bad file reports
  every problem (a 3-error file -> 3 lines, exit 1); schema exports and
  round-trips (jsonschema-validated against the hand-written example).

## Checks (must run and pass before you claim done)

  /home/jackson/miniforge3/envs/rmgpu/bin/rmgpu validate
    examples/minimal.yaml -> exit 0
  .../rmgpu run examples/minimal.yaml -> valid YAML on stdout
  pytest tests/test_cli.py -q -> all pass

## Pitfalls

- validate must NOT fail-fast - "report every problem at once" is
  the UX win (PLAN.md 12.1).
- `rmgpu run` does not execute chemistry yet - do not wire main.py; print +
  exit only.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-03/step-03: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-03/step-03 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-03-step-03-cli.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
