# job-00/step-02: Package skeleton + CLI stubs + test scaffolding

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

- Previous step (already done, its code is in the tree): job-00/step-01-env
- This step: job-00/step-02-skeleton
- Next step (do NOT start it): job-00/step-03-gate

## Goal

Create the rmgpu package skeleton exactly per ORIENTATION.md's "Files
you will create" layout, with CLI stubs and pytest scaffolding. Bare - later
steps fill the modules.

## Reference to read (this step's budget)

  /home/jackson/rmgpu/rmgpu/ORIENTATION.md  ("Files you will create")
  /home/jackson/rmgpu/RMG-Py/rmgpy/rmg/input.py  (2057 lines - READ TOC ONLY:
    the 41 DSL functions' names/signatures; job 03 ports the semantics. Do not
    read the bodies.)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- pyproject.toml: name rmgpu, version 0.1.0, entry point
  `rmgpu = rmgpu.cli:main`, deps from the env list; editable install works
  (`pip install -e .` in the env; record that).
- rmgpu/ with empty-but-valid __init__.py per subpackage:
  core/ molecule/ db/ ml/ kinetics/ pdep/ statmech/ reactor/ io/ schemas/
  tools/ sensitivity/ plugins/ (plugins/ and sensitivity/ exist as empty
  packages now so later steps never touch __init__ plumbing);
  rmgpu/__init__.py exposes __version__.
- rmgpu/cli.py: click group `rmgpu` with subcommands `run` (prints "not yet
  implemented"), `validate` (prints "not yet implemented"), `version` (works).
- tests/conftest.py: adds repo root to sys.path; fixtures:
  `example_dir` -> /home/jackson/rmgpu/RMG-Py/examples/rmg,
  `ref_db` -> /home/jackson/rmgpu/RMG-database,
  `rmgpy` -> /home/jackson/rmgpu/RMG-Py/rmgpy.
- gates/README.md: the gate contract - every job gate is gates/gate_NN.py,
  runs under the rmgpu env python, prints PASS/FAIL, exit 0/1, writes
  reports/job-NN.md. gates/baselines/ directory exists (empty).
- reports/ directory exists (empty).

## Checks (must run and pass before you claim done)

  /home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest tests/ -q
    -> passes (trivial test at least: version import)
  /home/jackson/miniforge3/envs/rmgpu/bin/python -m rmgpu.version (or
    `rmgpu version` via console script) -> 0.1.0
  rmgpu version -> 0.1.0

## Pitfalls

- Keep the skeleton BARE - do not pre-write chemistry code.
- The layout is load-bearing: later steps assume these exact subpackage
  names; do not rename.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-00/step-02: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-00/step-02 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-00-step-02-skeleton.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
