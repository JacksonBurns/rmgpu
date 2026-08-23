# job-06/step-03: main.py: the job driver + the iteration loop

Job: job-06 - Core/edge mechanism loop + torchdae reactor (first integration)
Prereq: jobs 01-05 done (all of: molecule, db, schema, estimators, recipes)
This step is part of that job. The job's overall goal:
The `rmgpu run` command actually works end to end (gas phase, constant T or T/P): load YAML -> build model from seed species -> estimate properties (libraries/ML) -> enumerate candidate reactions (recipes) -> simulate in torchdae -> screen by conversion -> grow/prune -> iterate to steady state -> write the output tree (PLAN.md 12.3), including the canonical `mechanism/core.yaml`. The moment of truth for the stack.
The job's gate (run by the job's final step):
gates/gate_06.py: 1. torchdae sub-gate: a stiff reference ODE (a 3-reaction Lindemann falloff toy system, or Van der Pol mu=10 as a non-chemistry control) integrated by torchdae vs a high-accuracy reference (RK45 tiny step) - max abs diff recorded (PLAN.md 13 risk 3: prove it before trusting it with mechanism growth). 2. `rmgpu run` on the imported `superminimal` (HPL kinetics, pdep off/stub): completes to steady state (or max_iter); iteration count + core/edge species+reaction counts recorded. 3. Parity vs RMG-Py (pdep OFF, same input; its output -> gates/baselines/superminimal/): |core_rmgpu - core_rmg| and |core+edge_rmgpu - (core+edge)_rmg| as sets. TARGET: core identical (or documented divergence with cause); edge within a small fraction. Any systematic divergence (a family always missing/extra): fix if tractable in this job, else record precisely. 4. Same for `c3h4` (the right size - NOT the GRI-scale example). 5. Output tree: every file in PLAN.md 12.3 exists and is valid (YAML parses, CSV columns right, core.yaml loads back via the schema and re-simulates the final iteration's profiles within tolerance). 6. provenance.yaml contains real hashes/versions (not placeholders).

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

- Previous step (already done, its code is in the tree): job-06/step-02-model
- This step: job-06/step-03-driver
- Next step (do NOT start it): job-06/step-04-output

## Goal

The job driver: load resolved YAML -> build everything -> run the
iteration loop to steady state. `rmgpu run` becomes real (execution, not
just validation) - but the output tree is still minimal here (step 5 lands
the full writer).

## Reference to read (this step's budget)

  RMG-Py/rmgpy/rmg/main.py  (2794 - run_rmg: the job driver, the
    iteration loop, the simulation scheduling, the steady-state criterion -
    port the orchestration; READ THE LOOP STRUCTURE, not every helper - the
    helpers are the modules this job wires)
  (everything: job-02 Databases, job-03 load_input, job-04 estimators,
    job-05 families, job-06/1 reactor, job-06/2 model)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/main.py:
    `run(input_path)`:
      load_input (job 03) -> resolved document;
      build Databases (job 02) from the database: block;
      build estimators (job 04) from the ml_estimator: block (checkpoint
      refs; if a checkpoint is missing on the box: the run proceeds with
      the estimators in BLOCKED state - every ML hit becomes a recorded
      coverage finding, the run completes, the report says what was blocked
      - NEVER a silent fallback and never a crash);
      build families (job 05) from the kinetics_families selection (the
      'default' set or a list; blocked families from job-05's list are
      skipped + recorded);
      build reactors (job 06/1) from the reactors: block;
      build the model (job-06/2) from the species: block (seed species:
      structures parsed via job-01 Molecule; seed mechanisms: a placeholder
      - the mechanism artifact loader lands in job 08/10 with the schema,
      for now seed species only + a documented note);
      run the iteration loop (enlarge/simulate/screen/prune per
      CoreEdgeReactionModel; max iterations / steady-state criterion per
      RMG's - the model block's tolerances).
    `rmgpu run input.yaml` (CLI, job 03) now EXECUTES (call main.run) and
    writes a MINIMAL output (the full tree is step 5): the mechanism
    (core/edge species+reactions) as a working YAML + the iteration count.
- tests/test_main_driver.py: `rmgpu run` on examples/minimal.yaml (job
  03's hand-written one - if it is too small to iterate, use the imported
  superminimal) runs to completion; the model is non-empty; the run is
  deterministic (two runs -> same core set, same counts).

## Checks (must run and pass before you claim done)

  /home/jackson/miniforge3/envs/rmgpu/bin/rmgpu run
    examples/minimal.yaml (or the imported superminimal) -> completes,
    prints the iteration count + core/edge counts
  pytest tests/test_main_driver.py -q -> all pass
  two consecutive runs: identical core species+reaction sets (determinism)

## Pitfalls

- The iteration loop's steady-state criterion is RMG's (consecutive
  iterations with no core change / the interrupt tolerance) - port it
  exactly; a wrong criterion changes the final mechanism.
- The pdep hook is STILL the HPL stub in this step - do not wire job 07 in
  early; the stub is documented so job 07 plugs in without touching the
  loop.
- Determinism: the loop must be deterministic (sorted reaction generation,
  no dict-order dependence) - the gate re-runs the example; a non-
  deterministic loop makes parity undecidable. Note any ordering you pinned
  in the report.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-06/step-03: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-06/step-03 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-06-step-03-driver.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
