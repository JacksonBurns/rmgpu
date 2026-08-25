# job-08/step-06: Job-08 gate (methods + isotope + observables + exports)

Job: job-08 - pdep MSC/RS/SLS + isotope + observables + diff/merge + exports
Prereq: job-07 done (CSE works + gated; the network/DoS core)
This step is part of that job. The job's overall goal:
Complete the pressure-dependence method set (MSC, RS, SLS - onto job 07's network/DoS core) and the analysis/interop tooling the parity suite (job 10) needs: isotope support, observables, model diff/merge, and the legacy-format exports (Chemkin read, Cantera YAML, RMS) from the canonical artifact.
The job's gate (run by the job's final step):
gates/gate_08.py: 1. MSC/RS/SLS parity: propane_branching (job 07's setup) with each of the 4 methods in rmgpu vs RMG-Py: per-reaction k(T,P) max relative diff + network-structure match (same tolerances as job 07). 2. Isotope: an isotope example from test/regression (inventory first) in rmgpu vs RMG-Py: species/thermo parity (the ZPE shifts) + mechanism-set parity. 3. Observables: on the c3h4 run (job 06), RMG-Py's reference observables vs rmgpu's: max abs/rel diff per observable type. 4. diff/merge: two job-06 artifacts (superminimal at iter N and N+1): the diff matches RMG-Py's diffmodels on the equivalent models; the merge round-trips (merge(A,B) contains both, no dupes). 5. Export round-trips: core.yaml -> chemkin.inp -> re-read (ckcsvparser) -> core.yaml: species/reaction sets equal, rates match 1e-8. core.yaml -> cantera/chem.yaml: load in Cantera, check species/reaction counts + a sample of NASA coeffs + a sample of Falloff params vs the artifact.

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

- Previous step (already done, its code is in the tree): job-08/step-05-exports
- This step: job-08/step-06-gate
- Next step (do NOT start it): the job gate for job-08 (its step file)

## Goal

Run the job-08 gate: the 4-method pdep parity, the isotope example,
the observables on c3h4, the diff/merge, the export round-trips. Close
job 08 - the parity suite (job 10) is unblocked.

## Reference to read (this step's budget)

  (no new reference reads)
  the isotope example: inventory test/regression for isotope inputs (the
    gate picks one)
  /home/jackson/rmgpu/RMG-Py/examples/rmg/c3h4/ (the observables reference)
    + propane_branching (the 4-method reference)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- gates/gate_08.py: the five checks (the 4-method pdep parity on
  propane_branching (per-reaction k(T,P) max rel diff + the network
  structure; same tolerances as job 07); the isotope example (species/
  thermo parity + the mechanism set); the observables on c3h4 (the max
  abs/rel diff per observable type vs RMG-Py's reference); the diff/merge
  (the two superminimal artifacts - the diff vs RMG-Py's diffmodels, the
  merge round-trip); the export round-trips (the Chemkin 1e-8, the
  Cantera load)).
- The RMG-Py references: the 4-method pdep (propane_branching, each
  method), the isotope example (its species/thermo + mechanism), the c3h4
  observables -> gates/baselines/job08/ (commit).
- reports/job-08.md: all five checks' numbers (real), any deviation +
  cause.
- The full tests/ suite green.
- If a method's k(T,P) is RED: the diagnosis is the method's port (step 1)
  - the core (job 07) is gated; fix the method, re-run.

## Checks (must run and pass before you claim done)

  /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_08.py
    -> the five checks recorded (target: job-07 tolerances on the
    methods, 1e-8 on the exports, the isotope + observables within the
    recorded tolerances)
  pytest tests/ -q -> all pass

## Pitfalls

- The 4 methods share job 07's core - a method RED is the method
  port (step 1), not the core; the core is gated (job 07).
- The isotope example: pick one that EXISTS in test/regression (inventory
  first) - if none is clean, use the smallest + document it.
- The exports are 1e-8 (the round-trip) - a larger diff is a format
  drift (the reader vs the writer) - pin it before job 10 (the parity
  suite reads the artifacts).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-08/step-06: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-08/step-06 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-08-step-06-gate.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
