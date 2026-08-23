# job-10/step-02: Battery chunk A (the small + filter/prune examples)

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

- Previous step (already done, its code is in the tree): job-10/step-01-harness
- This step: job-10/step-02-battery-a
- Next step (do NOT start it): job-10/step-03-battery-b

## Goal

Battery chunk A: the small examples + the filter/prune logic
examples (minimal, minimal_thermofilter, minimal_staged,
minimal_sensitivity (job 09), pruning_test, heptane-filterReactions).
Each via the harness (step 10/1); the INDEX updated.

## Reference to read (this step's budget)

  the gas-phase battery list (step 10/1's report) - chunk A:
    minimal, minimal_thermofilter, minimal_staged, minimal_sensitivity,
    pruning_test, heptane-filterReactions
  (the harness from step 10/1; the observables; the diffmodels)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- Run chunk A via scripts/parity_example.py (one example per
  sub-task - each is a session's work; the chunk is the step, the
  examples are its units):
    for each: the protocol (import, the RMG-Py reference, the rmgpu run,
    the parity report, the INDEX row).
  - The filter/prune examples (minimal_thermofilter, pruning_test,
    heptane-filterReactions) exercise the filter/prune logic (job 06/2) -
    a divergence here is a bookkeeping bug (the screen/prune rule), not an
    estimator issue - root-cause to the rule.
  - minimal_sensitivity (job 09): the sensitivity/uncertainty outputs are
    part of the parity (the covariance files) - the observable parity +
    the sensitivity agreement.
- The INDEX updated (the chunk's rows).
- Any systematic divergence (a cause recurring across the chunk's
  examples): fix the responsible module (job 04/05/06/07/08) - not the
  example - + re-run the affected examples.

## Checks (must run and pass before you claim done)

  the chunk A examples: reports/parity/<example>.md written + the
    INDEX rows filled (pass / pass-with-deviation / fail (cause))
  any systematic divergence: the fix commit + the re-run (the affected
    examples' INDEX rows updated)

## Pitfalls

- Divergences cluster by CAUSE (one bad family matcher, one
  estimator edge case, one pdep fit) - a recurring cause across the
  chunk is a module bug (fix the module, re-run), not per-example
  special-casing (special-casing an example is a parity violation).
- Do NOT tune the thresholds (toleranceKeepInEdge etc.) to force
  agreement - the imported YAML carries RMG's exact values (job 03).
- Do NOT lower the max-edge-species cap to force a match - parity means
  same input, same parameters.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-10/step-02: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-10/step-02 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-10-step-02-battery-a.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
