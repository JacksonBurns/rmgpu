# job-02/step-05: Transport/StatMech/Solvation facades + job-02 gate

Job: job-02 - Database layer via rmgdb
Prereq: job-01 done (Molecule for substructure lookups)
This step is part of that job. The job's overall goal:
ALL database I/O for rmgpu goes through rmgdb - the SQL/SQLite wrapper over RMG-database. No YAML-parsing of RMG-database files inside rmgpu (except data rmgdb does not cover yet, which must be documented, not silently worked around). Deliverables: rmgpu/db/ (loaders + facades), rmgpu/data/ (entries, rate-model math, thermo models, kinetics retrieval), rmgpu/kinetics/models.py (the rate registry that later jobs consume).
The job's gate (run by the job's final step):
gates/gate_02.py: 1. Round-trip count: for each library in a fixed list (recommended_ libraries.yml 'default' set + primaryThermoLibrary + training depository): entry count via rmgdb == entry count via RMG-Py loading the same library (RMG-Py script -> gates/baselines/db_counts.json). Must be EXACT. 2. Content hash: primaryThermoLibrary + one kinetics library: dump all entries (label/sorted-smiles/coeffs/T-bounds) to sorted YAML from BOTH rmgdb (via rmgpu) and RMG-Py; diff byte-identical after normalization. 3. Lookup parity: 25 random species from superminimal/c3h4: thermo lookup via rmgpu == via RMG-Py (same model type + coefficients, tol 1e-12). 4. Rate-model round-trip: 100 reactions sampled from kinetics.db: build the rmgpu rate model from stored params; k(300,1bar) and k(1000,1bar) match RMG-Py's evaluation of the same params (rel tol 1e-10). 5. rmgdb coverage gaps documented (report + STATUS decisions if they hit later jobs).

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

- Previous step (already done, its code is in the tree): job-02/step-04-kineticsdb
- This step: job-02/step-05-miscdb
- Next step (do NOT start it): the job gate for job-02 (its step file)

## Goal

The remaining minimal facades (transport, statmech, solvation -
each sized for its consuming job), then the job-02 gate.

## Reference to read (this step's budget)

  /home/jackson/rmgpu/rmgdb/standard/rmgdb/{transport,statmech,
    solvation}/schema.py  (the tables; each facade = what its consuming job
    needs: transport -> LJ sigma/epsilon + collision parameters (jobs 06/07);
    statmech -> group characteristic frequencies + libraries (job 07);
    solvation -> solvation groups/libraries (job 11))
  /home/jackson/rmgpu/rmgdb/demo.ipynb (any transport/statmech usage)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/db/loaders.py (extend): `TransportDB` (LJ params,
  collision data - the minimum job 06/07 needs), `StatMechDB` (group
  frequency data + statmech libraries - the minimum job 07 needs),
  `SolvationDB` (solvation groups/libraries - the minimum job 11 needs).
  Each: lookup API + entry counts + a docstring naming the consuming job.
- gates/gate_02.py: the full gate per the job definition (counts, content
  hash, lookup parity, rate-model round-trip, gap list).
- reports/job-02.md: all five check results with real numbers; the gap
  list (any rmgdb coverage gaps found, each with: what's missing, the
  workaround used (raw RMG-database file read, documented), and which job
  it hits).
- Update Databases aggregate to construct all facades from a config dict
  (library lists, family list - mirrors the YAML database: block of job 03).

## Checks (must run and pass before you claim done)

  /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_02.py ->
    PASS (or red with the failing check listed + cause)
  pytest tests/ -q -> all pass

## Pitfalls

- If rmgdb has a bug/gap: DO NOT fork it into rmgpu; query the raw
  RMG-database file for just that bit, document it, flag it for the user
  (STATUS decisions log).
- The content-hash check (gate 2) catches rmgdb data corruption - if it
  fails, that is an rmgdb bug to report, not an rmgpu bug.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-02/step-05: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-02/step-05 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-02-step-05-miscdb.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
