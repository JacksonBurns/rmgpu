# job-10/step-01: The parity harness + the INDEX + the first examples

Job: job-10 - Full gas-phase feature parity (the parity milestone)
Prereq: jobs 01-09 done + gated (esp. 04 thesis-test, 06, 07, 08)
This step is part of that job. The job's overall goal:
`rmgpu run` reproduces RMG-Py's gas-phase behavior across the standard example + regression battery, with parity measured per PLAN.md 12.5: diff the canonical mechanism/core.yaml artifacts (NOT the Chemkin round-trip) + the observable-level comparison (job 08). This job adds NO features - it runs the battery + drives the remaining gas- phase examples to green. A CAMPAIGN job: it may span many sessions (the steps below are chunks of the battery; each is a session).
The job's gate (run by the job's final step):
gates/gate_10.py: reads reports/parity/INDEX.md + the per-example reports; asserts: all gas-phase battery examples are `pass` or `pass- with-deviation` (documented); the `fail` count is 0 (or each `fail` has an accepted, documented cause in the STATUS decisions log - user- approved). THE parity milestone (PLAN.md 10 Phase 4): a green gate means "rmgpu reproduces RMG's gas-phase mechanism generation" - the headline result of the PoC. Be rigorous; do not cherry-pick examples.

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
- This step: job-10/step-01-harness
- Next step (do NOT start it): job-10/step-02-battery-a

## Goal

The parity harness: the per-example protocol as a script (the unit
of work), the parity INDEX (the matrix maintained across sessions), and
the first examples (superminimal + c3h4 re-confirmed with pdep ON).

## Reference to read (this step's budget)

  (the job-06/07 gates' machinery - the RMG-Py reference scripts,
    the artifact diff (job 08/4's diffmodels), the observables (job
    08/3))
  the gas-phase battery: the 47 legacy inputs (job 03's inventory) minus
    the liquid/surface/mb_sampled (jobs 11/12) - the list is in the job
    brief
  /home/jackson/rmgpu/RMG-Py/examples/rmg/{superminimal,c3h4}/ (the first
    two)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- scripts/parity_example.py: the per-example protocol (the unit of
  work), a script:
    (1) import the example's input.py -> YAML (job 03's importer - if a
    documented IMPORT-NOTE case, resolve the note - the example must run);
    (2) run RMG-Py on the original input.py (env rmg_env, same machine,
    same DB) -> gates/baselines/<example>/ (the final mechanism +
    profiles);
    (3) run rmgpu on the imported YAML -> the output tree;
    (4) the parity report (reports/parity/<example>.md): the core species
    set |D| (exact-match target), the core reaction set |D|, the edge
    |D| (small fraction ok, list the stragglers), the observables (job
    08) max abs/rel diff per type, any systematic divergence (root-
    caused: family missing? estimator? pdep fit? screening?) + the
    responsible module.
- reports/parity/INDEX.md: the parity matrix (one row per example:
  example | core species D | core rxn D | edge D | observables | status
  | notes) - the status values (pending | pass | pass-with-deviation
  (documented) | fail (blocked, cause recorded)).
- The first examples: superminimal + c3h4 (re-confirm job-06's numbers
  WITH pdep ON - job 07's pdep changes the rates; the parity target is
  vs RMG-Py pdep-on now, not the HPL run).
- The battery list (the gas-phase set, prioritized per the job brief) -
  recorded in the report (the campaign's work list).

## Checks (must run and pass before you claim done)

  scripts/parity_example.py runs on superminimal + c3h4 (the
    reports written, the INDEX rows filled)
  the two examples' parity: the core set D recorded (target: identical
    or documented)

## Pitfalls

- The parity reference is RMG-Py with the SAME settings (pdep ON
  for both now - job 07) - a pdep-state mismatch is a category error.
- The IMPORT-NOTE cases (job 03): an example that had a documented note
  must have it RESOLVED to run (the note is not a parity excuse) - if the
  note is unresolvable, the example is BLOCKED (recorded, user decision),
  not silently skipped.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-10/step-01: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-10/step-01 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-10-step-01-harness.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
