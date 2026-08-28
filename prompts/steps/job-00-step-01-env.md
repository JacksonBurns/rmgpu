# job-00/step-01: Conda env rmgpu (all deps, rmgdb, checkpoint locations)

Job: job-00 - Environment, package skeleton, test scaffolding
Prereq: none (first job)
This step is part of that job. The job's overall goal:
Set up the foundation every later job builds on: the `rmgpu` conda env, the package skeleton, test tooling, and the gate harness. No chemistry yet.
The job's gate (run by the job's final step):
gates/gate_00.py + tests/test_smoke.py all pass: - every rmgpu subpackage imports; `rmgpu version` prints 0.1.0 - `import rdkit, torch, chemprop, cantera, chemicals, fluids, thermo, pint, sqlalchemy, torchdae, pydantic` all work; torch is the CUDA build (torch.cuda.is_available() is True) - the CheMeleon checkpoint paths used by RMG-Py's ml_estimator DSL are LOCATED and recorded (exact paths, formats, output dims) in the report - not made to work yet (job 04), but the env must be able to load one - rmgdb is installed and importable; install method recorded

First step of this job: skim /home/jackson/rmgpu/rmgpu/ORIENTATION.md once
(what stays/goes/external; the master-equation data path; conventions). Later
steps of this job do not need it - everything they need is in their own file.

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

- Previous step (already done, its code is in the tree): (this is the first step of the job)
- This step: job-00/step-01-env
- Next step (do NOT start it): job-00/step-02-skeleton

## Goal

Create the `rmgpu` conda environment (python 3.11) with every
dependency, install rmgdb, and locate the CheMeleon checkpoint files so job 04
knows exactly what it is consuming.
NOTE (2026-08-27 plan update): the two deployed checkpoints are now VENDORED
in this repo's top-level `models/` directory (chemeleon_thermo_662946.ckpt,
chemprop_kinetics_662946.ckpt) - see PLAN.md 3b and the updated checkpoint
inventory in reports/job-00.md. This step's original task (locating
checkpoints referenced by RMG-Py's ml_estimator DSL) predates that.
The env must hold: torch (CUDA build - if pip resolves a CPU wheel, fix the
index), numpy, scipy, rdkit, pint, chemprop, lightning, cantera, chemicals,
fluids, thermo, torchdae, sqlalchemy, polars, pydantic, click, pytest,
pytest-cov (lightning is REQUIRED: both vendored checkpoints load/predict
through pytorch-lightning).
torchdae on PyPI is 0.1.1; record whatever version lands - the API surface we
need is BDF1/BDF2/TR-BDF2/Radau-IIA + index reduction + adjoint (PLAN.md 5).

## Reference to read (this step's budget)

  /home/jackson/rmgpu/rmgdb/README.md and standard/rmgdb/*/schema.py
    (skim - so the skeleton has the right stubs; job 02 reads them in depth)
  /home/jackson/rmgpu/RMG-Py/rmgpy/ml/estimator.py  (182 lines - the OLD
    estimator; read to learn the checkpoint layout it consumes: hf298_path,
    s298_cp_path, uncertainty cutoffs. It is being REPLACED in job 04; we are
    only locating what checkpoints it references.)
  /home/jackson/rmgpu/RMG-Py/examples/rmg/minimal_ml/input.py  (how the
    ml_estimator DSL points at checkpoints)
  /home/jackson/rmgpu/chemprop_example/predicting.ipynb  (190 lines - the NEW
    reference: models.MPNN.load_from_checkpoint(checkpoint_path),
    featurizers.SimpleMoleculeMolGraphFeaturizer, data.MoleculeDataset,
    data.build_dataloader, pl.Trainer(accelerator=...).predict)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- `~/miniforge3/envs/rmgpu` exists; python 3.11; all packages above
  import under /home/jackson/miniforge3/envs/rmgpu/bin/python.
- rmgdb installed (follow its README: editable install of the `standard`
  package and the data builder; built SQLite at
  /home/jackson/rmgpu/rmgdb/db/{thermo,kinetics,transport,solvation,statmech}.db
  must be present - if a .db is missing, run its documented build step and
  record that).
- Checkpoint inventory (HISTORICAL, from the original step-01 run): for every
  checkpoint referenced by RMG-Py's ml_estimator DSL / minimal_ml example
  (CheMeleon thermo: Hf298 model, S298+Cp model), record: exact path, file
  format (.ckpt / torchscript / other), and if loadable with the
  chemprop_example pattern, the model's input featurizer type + output dims.
  SEE THE UPDATED CHECKPOINT INVENTORY IN reports/job-00.md: the two deployed
  checkpoints are now vendored in this repo's models/ directory (PLAN.md 3b).
- A one-off env-verification script (keep it: scripts/check_env.py) that
  imports everything and prints versions - later steps re-run it.

## Checks (must run and pass before you claim done)

  /home/jackson/miniforge3/envs/rmgpu/bin/python scripts/check_env.py
  -> all imports OK, torch.cuda.is_available() True, versions printed
  rmgdb importable; `import rmgdb; print(rmgdb.__version__ if hasattr(rmgdb,'__version__') else 'ok')`
  at least one CheMeleon checkpoint located; if loadable via the
  chemprop_example pattern, a 1-molecule predict runs (record output)

## Pitfalls

- Never install into system python or the active venv.
- If torch lands CPU-only, fix the wheel index and RE-VERIFY
  torch.cuda.is_available() before finishing.
- Do not pre-write chemistry code - skeleton only (next step).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-00/step-01: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-00/step-01 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-00-step-01-env.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
