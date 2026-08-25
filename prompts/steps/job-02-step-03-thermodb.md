# job-02/step-03: ThermoDB facade + thermo models (Wilhoit/NASA7)

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

- Previous step (already done, its code is in the tree): job-02/step-02-falloff
- This step: job-02/step-03-thermodb
- Next step (do NOT start it): job-02/step-04-kineticsdb

## Goal

Build the thermo side of the database layer: the ThermoDB facade over
rmgdb, the data-only entry classes, and the thermo MODEL math (Wilhoit,
NASA7) - NOT estimation (that is job 04, ML, and the ONLY estimator).

## Reference to read (this step's budget)

  /home/jackson/rmgpu/rmgdb/README.md + standard/rmgdb/thermo/schema.py
    (the thermo tables)
  /home/jackson/rmgpu/rmgdb/demo.ipynb  (canonical query usage - READ IT)
  RMG-Py/rmgpy/thermo/wilhoit.pyx  (927 - Wilhoit: Hf298/S298/Cp(T)
    polynomial; port in numpy - RMG-specific and small)
  RMG-Py/rmgpy/thermo/nasa.pyx  (445 - NASA7; decision per PLAN.md 5: thin
    numpy eval or Cantera - implement, record the choice)
  RMG-Py/rmgpy/data/thermo.py  (3099 - the entry class + the GA/HBI
    estimation is DELETED; port only the entry/model semantics)
  RMG-database/input/recommended_libraries.yml (the library sets)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/data/entries.py: data-only classes matching RMG semantics:
  ThermoEntry (Wilhoit or NASA7 coefficients + T bounds + reference),
  KineticsEntry (one rate model + reference) with serialization hooks for
  the mechanism artifact (job 06 defines the schema; keep it plain data).
- rmgpu/data/thermo.py (thin): the thermo MODEL side: Wilhoit (port in
  numpy), NASA7 (numpy or Cantera per the decision); get_heat_capacity /
  get_enthalpy / get_entropy / get_cp, T-dependent, SI units.
- rmgpu/db/loaders.py: `ThermoDB` facade over rmgdb: entry lookup by
  structure (substructure match against library entries via job-01
  Molecule + rmgdb thermo tables), label lookup, "is this in library X",
  entry counts. `Databases` aggregate skeleton (constructed from library
  lists; caching) - other DBs added by later steps.
- tests/test_thermodb.py: lookup on 25 species (superminimal/c3h4) matches
  RMG-Py (same model type + coefficients, 1e-12); entry counts for
  primaryThermoLibrary match RMG-Py (script); Wilhoit Cp(T)/H(T)/S(T) vs
  RMG-Py on a coefficient set (hand values + a real library entry).

## Checks (must run and pass before you claim done)

  pytest tests/test_thermodb.py -q -> all pass
  counts: rmgpu ThermoDB entry count == RMG-Py load count (primaryThermo
  Library) - recorded

## Pitfalls

- DO NOT port group additivity / HBI / any "estimation" (job 04 is
  the only estimator - PLAN.md 3). If you catch yourself porting GA, stop.
- rmgdb is the single I/O path - if a library does not resolve via rmgdb,
  document the gap (do not fork a YAML reader silently); the gate's gap list
  is the record.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-02/step-03: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-02/step-03 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-02-step-03-thermodb.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
