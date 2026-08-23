# job-08/step-03: Observables + regression comparison

Job: job-08 - pdep MSC/RS/SLS + isotope + observables + diff/merge + exports
Prereq: job-07 done (CSE works + gated; the network/DoS core)
This step is part of that job. The job's overall goal:
Complete the pressure-dependence method set (MSC, RS, SLS - onto job 07's network/DoS core) and the analysis/interop tooling the parity suite (job 10) needs: isotope support, observables, model diff/merge, and the legacy-format exports (Chemkin read, Cantera YAML, RMS) from the canonical artifact.
The job's gate (run by the job's final step):
gates/gate_08.py: 1. MSC/RS/SLS parity: propane_branching (job 07's setup) with each of the 4 methods in rmgpu vs RMG-Py: per-reaction k(T,P) max relative diff + network-structure match (same tolerances as job 07). 2. Isotope: an isotope example from test/regression (inventory first) in rmgpu vs RMG-Py: species/thermo parity (the ZPE shifts) + mechanism-set parity. 3. Observables: on the c3h4 run (job 06), RMG-Py's reference observables vs rmgpu's: max abs/rel diff per observable type. 4. diff/merge: two job-06 artifacts (superminimal at iter N and N+1): the diff matches RMG-Py's diffmodels on the equivalent models; the merge round-trips (merge(A,B) contains both, no dupes). 5. Export round-trips: core.yaml -> chemkin.inp -> re-read (ckcsvparser) -> core.yaml: species/reaction sets equal, rates match 1e-8. core.yaml -> cantera/chem.yaml: load in Cantera, check species/reaction counts + a sample of NASA coeffs + a sample of Falloff params vs the artifact.

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

- Previous step (already done, its code is in the tree): job-08/step-02-isotopes
- This step: job-08/step-03-observables
- Next step (do NOT start it): job-08/step-04-diffmerge

## Goal

Observables + the regression comparison - the tool job 10's parity
gate calls: the observable definitions (species conversion, time-to-x,
profile integrals) + the comparison against RMG-Py's reference.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/tools/observablesregression.py  (419 - the
    observable definitions (species conversion, time-to-x, profile
    integrals) + the regression comparison - port the semantics)
  (job-06's Profiles object - the observable computes over it)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/tools/observables.py:
    the observable types (species conversion, time-to-x, profile
    integrals - the definitions from observablesregression.py) computed
    over a run's Profiles (job 06);
    the regression comparison (the observable-vs-reference comparison -
    the max abs/rel diff per observable type) - the function job 10's
    gate calls.
- tests/test_observables.py: a hand-built profile (a known conversion
  curve): the observables (conversion, time-to-x, the integral) are
  correct (hand-computed values); the regression comparison on a synthetic
  reference (the diff is exact).

## Checks (must run and pass before you claim done)

  pytest tests/test_observables.py -q -> all pass
  the observables on a known profile: hand-computed values match

## Pitfalls

- The observables are what job 10's parity gate compares (the
  mechanism-set parity is the structure; the observables are the
  dynamics) - a wrong observable is a false parity result; the
  hand-computed values (this step) pin them before the gate.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-08/step-03: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-08/step-03 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-08-step-03-observables.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
