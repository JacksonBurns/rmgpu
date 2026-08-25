# job-10: Full gas-phase feature parity (the parity milestone)

Status of this file: a JOB BRIEF, not a task. Do not try to "do this job"
in one session - it is decomposed into the steps below, one session each.
The human reads STATUS.md's NEXT pointer; this file is the map of the job.

## Goal

`rmgpu run` reproduces RMG-Py's gas-phase behavior across the standard example + regression battery, with parity measured per PLAN.md 12.5: diff the canonical mechanism/core.yaml artifacts (NOT the Chemkin round-trip) + the observable-level comparison (job 08). This job adds NO features - it runs the battery + drives the remaining gas- phase examples to green. A CAMPAIGN job: it may span many sessions (the steps below are chunks of the battery; each is a session).

## Prereq

jobs 01-09 done + gated (esp. 04 thesis-test, 06, 07, 08)

## Steps (strictly sequential; one fresh human-started session each)

  step 01  prompts/steps/job-10-step-01-harness.md  The parity harness + the INDEX + the first examples
  step 02  prompts/steps/job-10-step-02-battery-a.md  Battery chunk A (the small + filter/prune examples)
  step 03  prompts/steps/job-10-step-03-battery-b.md  Battery chunk B (the oxidation + pdep examples)
  step 04  prompts/steps/job-10-step-04-battery-c.md  Battery chunk C (the big regression suite)
  step 05  prompts/steps/job-10-step-05-gate.md  Job-10 gate (the parity milestone)

## The job gate

Run by the final step's session (gates/gate_10.py, report
reports/job-10.md):

gates/gate_10.py: reads reports/parity/INDEX.md + the per-example reports; asserts: all gas-phase battery examples are `pass` or `pass- with-deviation` (documented); the `fail` count is 0 (or each `fail` has an accepted, documented cause in the STATUS decisions log - user- approved). THE parity milestone (PLAN.md 10 Phase 4): a green gate means "rmgpu reproduces RMG's gas-phase mechanism generation" - the headline result of the PoC. Be rigorous; do not cherry-pick examples.

## When the job is done

The final step's report (reports/job-10.md) has the gate result, the
job table row is `done` (or `blocked` with the cause), and the NEXT pointer
in STATUS.md targets job-11's first step (if there is a next job).
