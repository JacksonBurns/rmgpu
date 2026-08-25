# job-03/step-05: Job-03 gate (lossless import of 47 examples)

Job: job-03 - YAML input schema + CLI + legacy .py importer
Prereq: job-01 done (Molecule for structure parsing). job-02 not strictly required, but the database: block references library names from it.
This step is part of that job. The job's overall goal:
Replace RMG's "execute a Python file to configure the run" with a declarative, schema-validated YAML document + a small CLI + a lossless importer for the legacy .py DSL. This is the user-facing front door. Deliverables: rmgpu/schemas/input.py (pydantic), rmgpu/cli.py (click), rmgpu/importer/legacy.py (ast-based).
The job's gate (run by the job's final step):
gates/gate_03.py: 1. DSL inventory: 47 legacy input.py files x functions used (report table). 2. Import: N of 47 files import to schema-valid YAML with zero dropped values; the rest have documented IMPORT-NOTEs. Target: N == 47. 3. `rmgpu validate` on all 47 imported YAMLs: all pass. 4. `rmgpu run minimal.yaml` (the imported minimal example) prints the resolved document. 5. JSON schema exports and validates a hand-written minimal input.yaml (the example from PLAN.md 12.2).

## Context (invariant every step)

- You are a FRESH human-started session; you have no other context. This file plus
  what it points at is everything you need.
- Work in /home/jackson/rmgpu/rmgpu (git, branch main). Reference repos (read-
  only, at /home/jackson/rmgpu): RMG-Py, RMG-database, rmgdb, chemprop_example.
- Interpreter: always the env's python directly,
  /home/jackson/miniforge3/envs/rmgpu/bin/python (created in job 00).
  NEVER install into system python or the active venv.
- ONE session at a time; this session MUST NOT spawn subagents. Single heavy
  GPU task at a time on this machine; never kill processes you did not start.
- No Cython, no numba, no QM, no Arkane, no fallback estimators, no second
  reactor backend. SI units internally (J, K, Pa, mol) via pint.
- The existing Chemprop-based estimators in RMG-Py (rmgpy/ml/estimator.py)
  are being REPLACED by rmgpu's new ones - they are reference-only (layout,
  cutoffs, DSL wiring). rmgpu's estimators are re-implemented per
  /home/jackson/rmgpu/chemprop_example/predicting.ipynb. See README.md,
  "The ML estimators are NEW".

## Where you are

- Previous step (already done, its code is in the tree): job-03/step-04-legacy
- This step: job-03/step-05-gate
- Next step (do NOT start it): the job gate for job-03 (its step file)

## Goal

Run the job-03 gate: import all 47 legacy inputs, validate them,
diff against the ast ground truth, write the report. Target: N == 47 with
zero dropped values.

## Reference to read (this step's budget)

  (no new reference reads)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- tests/test_importer.py: for every one of the 47 legacy input.py
  files: import to YAML, re-parse against the schema (must validate), diff
  the resolved document against scripts/legacy_dump.py's JSON (structure
  equality; values only; ordering where the DSL is order-sensitive must be
  preserved).
- gates/gate_03.py: runs the inventory + import + validate + diff; the
  report gets: the 47x-functions inventory table; N of 47 lossless; the
  IMPORT-NOTE list (every file with notes + what is noted); validate
  results; the hand-written example check.
- reports/job-03.md: all of the above with real numbers.
- Files that cannot be fully imported: allowed ONLY with a documented
  IMPORT-NOTE and the schema-validatable subset still matching; list every
  such file.

## Checks (must run and pass before you claim done)

  /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_03.py
    (record N/47; target 47 - a miss is recorded with each file's cause;
    the job is GREEN at N==47 or with every miss user-approved in STATUS)

## Pitfalls

- A missed value (a DSL feature the importer drops) is a gate
  failure, not a "known limitation" - PLAN.md 13 risk 8: the importer must
  cover all 41 functions or fail loudly, never silently default.
- The diff is structure equality on VALUES - formatting/keys that are
  schema-normalized are fine; dropped or altered values are not.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-03/step-05: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-03/step-05 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-03-step-05-gate.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
