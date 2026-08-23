# job-00/step-03: Smoke test + job-00 gate

Job: job-00 - Environment, package skeleton, test scaffolding
Prereq: none (first job)
This step is part of that job. The job's overall goal:
Set up the foundation every later job builds on: the `rmgpu` conda env, the package skeleton, test tooling, and the gate harness. No chemistry yet.
The job's gate (run by the job's final step):
gates/gate_00.py + tests/test_smoke.py all pass: - every rmgpu subpackage imports; `rmgpu version` prints 0.1.0 - `import rdkit, torch, chemprop, cantera, chemicals, fluids, thermo, pint, sqlalchemy, torchdae, pydantic` all work; torch is the CUDA build (torch.cuda.is_available() is True) - the CheMeleon checkpoint paths used by RMG-Py's ml_estimator DSL are LOCATED and recorded (exact paths, formats, output dims) in the report - not made to work yet (job 04), but the env must be able to load one - rmgdb is installed and importable; install method recorded

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

- Previous step (already done, its code is in the tree): job-00/step-02-skeleton
- This step: job-00/step-03-gate
- Next step (do NOT start it): the job gate for job-00 (its step file)

## Goal

Write the smoke test and the job-00 gate script, run them, and write
the job-00 report. This closes job 00.

## Reference to read (this step's budget)

  (no new reference reads; use reports from steps 01-02)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- tests/test_smoke.py: import rmgpu + every subpackage; assert
  version 0.1.0; `rmgpu version` via subprocess; assert the full import list
  from step 01 works; assert torch.cuda.is_available().
- gates/gate_00.py: runs the smoke checks as a script (exit 0/1, PASS/FAIL
  line), writes reports/job-00.md.
- reports/job-00.md: env versions (pip freeze | grep -E
  'torch|rdkit|chemprop|cantera|chemicals|fluids|thermo|torchdae|pint|
  sqlalchemy|polars|pydantic'), checkpoint inventory from step 01 (paths,
  formats, loadability), rmgdb install method, pytest summary line.

## Checks (must run and pass before you claim done)

  /home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest tests/ -q
    -> all pass
  /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_00.py -> PASS

## Pitfalls

- A red gate here is a BLOCKER for the whole project: fix env
  issues (wrong torch wheel, missing rmgdb build) before declaring done.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-00/step-03: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-00/step-03 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-00-step-03-gate.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
