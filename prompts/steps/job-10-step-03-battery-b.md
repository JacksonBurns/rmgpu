# job-10/step-03: Battery chunk B (the oxidation + pdep examples)

Job: job-10 - Full gas-phase feature parity (the parity milestone)
Prereq: jobs 01-09 done + gated (esp. 04 thesis-test, 06, 07, 08)
This step is part of that job. The job's overall goal:
`rmgpu run` reproduces RMG-Py's gas-phase behavior across the standard example + regression battery, with parity measured per PLAN.md 12.5: diff the canonical mechanism/core.yaml artifacts (NOT the Chemkin round-trip) + the observable-level comparison (job 08). This job adds NO features - it runs the battery + drives the remaining gas- phase examples to green. A CAMPAIGN job: it may span many sessions (the steps below are chunks of the battery; each is a session).
The job's gate (run by the job's final step):
gates/gate_10.py: reads reports/parity/INDEX.md + the per-example reports; asserts: all gas-phase battery examples are `pass` or `pass- with-deviation` (documented); the `fail` count is 0 (or each `fail` has an accepted, documented cause in the STATUS decisions log - user- approved). THE parity milestone (PLAN.md 10 Phase 4): a green gate means "rmgpu reproduces RMG's gas-phase mechanism generation" - the headline result of the PoC. Be rigorous; do not cherry-pick examples.

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

- Previous step (already done, its code is in the tree): job-10/step-02-battery-a
- This step: job-10/step-03-battery-b
- Next step (do NOT start it): job-10/step-04-battery-c

## Goal

Battery chunk B: the oxidation + pdep + dynamics examples (ethane-
oxidation, propane_branching (pdep - job 07's example, re-confirmed in
the campaign), ch3no2, nox_transitory_edge, minimal_dynamics). Via the
harness; the INDEX updated.

## Reference to read (this step's budget)

  the gas-phase battery list (step 10/1's report) - chunk B:
    ethane-oxidation, propane_branching, ch3no2, nox_transitory_edge,
    minimal_dynamics
  (the harness; the observables; the diffmodels)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- Run chunk B via scripts/parity_example.py (the same protocol):
    the oxidation examples (ethane-oxidation) are the first "real"
    mechanisms (bigger than c3h4) - the parity is the core set (the
    exact-match target) + the observables.
  - propane_branching: re-confirm in the CAMPAIGN (job 07 gated it
  standalone; the campaign re-runs it with the full stack - a regression
  here (job 08/09 changed something) is a real finding - root-cause to
  the changed module).
  - ch3no2 / nox_transitory_edge: the N-chemistry (the families + the
  estimators on N-species) - a divergence is likely an estimator coverage
  gap (job 04) or a family (job 05) - root-cause.
  - minimal_dynamics: the dynamics (the reactor + the termination) - a
    divergence is a reactor/termination bug (job 06/1).
- The INDEX updated.
- Any systematic divergence: fix the responsible module + re-run.

## Checks (must run and pass before you claim done)

  the chunk B examples: reports written + the INDEX rows filled
  any systematic divergence: the fix + the re-run

## Pitfalls

- The oxidation examples are the first where the ML estimators
  (job 04) are stressed (more species, the intermediates) - a coverage gap
  (the MLCoverageError findings) is a FINDING (recorded in the parity
  report + the STATUS), not a parity "failure" to hide (the no-fallback
  design: a gap is reported, PLAN.md 13 risk 1/2).
- propane_branching in the campaign is a REGRESSION check (job 07 gated
  it) - if it now disagrees with its own gate, something in job 08/09
  regressed it - find it.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-10/step-03: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-10/step-03 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-10-step-03-battery-b.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
