# job-10/step-05: Job-10 gate (the parity milestone)

Job: job-10 - Full gas-phase feature parity (the parity milestone)
Prereq: jobs 01-09 done + gated (esp. 04 thesis-test, 06, 07, 08)
This step is part of that job. The job's overall goal:
`rmgpu run` reproduces RMG-Py's gas-phase behavior across the standard example + regression battery, with parity measured per PLAN.md 12.5: diff the canonical mechanism/core.yaml artifacts (NOT the Chemkin round-trip) + the observable-level comparison (job 08). This job adds NO features - it runs the battery + drives the remaining gas- phase examples to green. A CAMPAIGN job: it may span many sessions (the steps below are chunks of the battery; each is a session).
The job's gate (run by the job's final step):
gates/gate_10.py: reads reports/parity/INDEX.md + the per-example reports; asserts: all gas-phase battery examples are `pass` or `pass- with-deviation` (documented); the `fail` count is 0 (or each `fail` has an accepted, documented cause in the STATUS decisions log - user- approved). THE parity milestone (PLAN.md 10 Phase 4): a green gate means "rmgpu reproduces RMG's gas-phase mechanism generation" - the headline result of the PoC. Be rigorous; do not cherry-pick examples.

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

- Previous step (already done, its code is in the tree): job-10/step-04-battery-c
- This step: job-10/step-05-gate
- Next step (do NOT start it): the job gate for job-10 (its step file)

## Goal

Run the job-10 gate: the battery is green (all gas-phase examples
pass or pass-with-deviation; the fail count 0 or user-approved). THE
parity milestone - the headline result of the PoC.

## Reference to read (this step's budget)

  reports/parity/INDEX.md (the matrix - the source of truth)
    + the per-example reports
  (the gate script reads the INDEX + the reports)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- gates/gate_10.py: reads reports/parity/INDEX.md + the per-
  example reports; asserts: all gas-phase battery examples are `pass` or
  `pass-with-deviation` (documented); the `fail` count is 0 (or each
  `fail` has an accepted, documented cause in the STATUS decisions log -
  user-approved).
- reports/job-10.md: the full INDEX.md inline + the summary (N pass, N
  pass-with-deviation (list + a one-line cause each), N fail (list +
  cause)) + the wall-time (the campaign's total + the GPU payoff).
- If the battery is NOT green: the gate is RED (the remaining examples
  listed + their causes) - the job is NOT done; the campaign continues
  (a new session picks up the remaining examples - the INDEX is the
  state). A green gate flips the job-10 row to done.
- A red gate at this point (examples still pending/fail) is EXPECTED for
  a campaign job - the gate re-runs each session; it goes green when the
  battery is done.

## Checks (must run and pass before you claim done)

  /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_10.py
    -> the battery status (green when all pass/pass-with-deviation; red
    with the remaining list otherwise)
  the full tests/ suite green

## Pitfalls

- The parity milestone is the headline (PLAN.md 10 Phase 4) - a
  green gate means "rmgpu reproduces RMG's gas-phase mechanism
  generation". Be rigorous: no cherry-picking (the INDEX is the record;
  the gate reads it).
- A `fail` with a user-approved cause (the STATUS decisions log) is
  allowed (the gate checks the log) - an unapproved fail is a RED gate.
- The campaign spans sessions - the INDEX + the per-example reports are
  the state; a new session reads the INDEX, picks the next pending
  example, runs it, updates the INDEX. The NEXT pointer (STATUS) targets
  the next pending example until the battery is green.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-10/step-05: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-10/step-05 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-10-step-05-gate.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
