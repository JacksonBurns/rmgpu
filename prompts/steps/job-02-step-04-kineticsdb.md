# job-02/step-04: KineticsDB facade + family-definition storage

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

- Previous step (already done, its code is in the tree): job-02/step-03-thermodb
- This step: job-02/step-04-kineticsdb
- Next step (do NOT start it): job-02/step-05-miscdb

## Goal

Build the kinetics side: KineticsDB facade (libraries + depository),
the kinetics retrieval skeleton, and - critically - investigate how family
definitions (recipes, templates, rate rules) are stored in rmgdb. Job 05
depends on that answer.

## Reference to read (this step's budget)

  /home/jackson/rmgpu/rmgdb/standard/rmgdb/kinetics/schema.py  (the
    kinetics tables - query kinetics.db empirically: does it store family
    recipes/templates/rate rules? document the schema you find)
  /home/jackson/rmgpu/rmgdb/demo.ipynb (kinetics usage)
  RMG-Py/rmgpy/data/kinetics/kineticsdata.pyx (already read in job 02/01?
    if not: the storage semantics)
  RMG-Py/rmgpy/data/kinetics/family.py  (4883 - READ TOC + the
    Family/FamilyDict loading path ONLY: how families load recipes/templates
    from the DB; job 05 reads the recipe engine itself)
  RMG-database/input/kinetics/families/<Name>/ (the family.py files -
    inspect 2-3: R_H_Abstraction, H_ABstraction - recipes as data-ish
    Python)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/db/loaders.py (extend): `KineticsDB` facade: entry lookup
  by reaction (substructure match), label lookup, library membership, entry
  counts, depository access (the 'training' depository is just a library of
  curated data in rmgpu - it feeds rate-rule TRAINING, a deleted feature;
  treat it as a library).
- rmgpu/data/kinetics.py (thin): retrieval logic: given a reaction, check
  libraries (substructure match), else the ML resolver (job 04 - stub with
  clear NotImplemented + TODO(job-04) for now; libraries override ML).
- FAMILY DEFINITION STORAGE: document in the report how rmgdb stores (or
  does not store) family recipes/templates/rate rules (query the schema +
  a sample family). If families are NOT in rmgdb: record the loading
  strategy decision for job 05 (controlled parse of the family.py files -
  they are data-ish Python; the decision + the parser home go in the
  report; job 05/step-4 implements it).
- tests/test_kineticsdb.py: entry counts per library vs RMG-Py (script);
  100-reaction rate-model round-trip is the GATE's (step 5) - here:
  lookup-parity on 25 reactions (species sets + rate model type match).

## Checks (must run and pass before you claim done)

  pytest tests/test_kineticsdb.py -q -> all pass
  family-storage investigation: a one-paragraph answer in the report
  (rmgdb stores families: YES/NO + what exactly + job-05 strategy)

## Pitfalls

- The depository is NOT a training data path in rmgpu (rate-rule
  training is deleted) - document that to prevent a later session "fixing"
  it back.
- Family storage is the key open question for job 05 - a vague answer here
  costs a session later. Be empirical: query the db, cite the tables.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-02/step-04: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-02/step-04 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-02-step-04-kineticsdb.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
