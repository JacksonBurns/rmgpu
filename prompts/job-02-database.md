# job-02: Database layer via rmgdb

Status of this file: a JOB BRIEF, not a task. Do not try to "do this job"
in one session - it is decomposed into the steps below, one session each.
The coordinator (see README.md, "Session vs step") picks the next step from
STATUS.md's NEXT pointer; this file is the map of the job.

## Goal

ALL database I/O for rmgpu goes through rmgdb - the SQL/SQLite wrapper over RMG-database. No YAML-parsing of RMG-database files inside rmgpu (except data rmgdb does not cover yet, which must be documented, not silently worked around). Deliverables: rmgpu/db/ (loaders + facades), rmgpu/data/ (entries, rate-model math, thermo models, kinetics retrieval), rmgpu/kinetics/models.py (the rate registry that later jobs consume).

## Prereq

job-01 done (Molecule for substructure lookups)

## Steps (strictly sequential; one fresh subagent session each)

  step 01  prompts/steps/job-02-step-01-arrhenius.md  Rate models: Arrhenius family + registry base
  step 02  prompts/steps/job-02-step-02-falloff.md  Rate models: falloff, Chebyshev, Marcus, tunneling
  step 03  prompts/steps/job-02-step-03-thermodb.md  ThermoDB facade + thermo models (Wilhoit/NASA7)
  step 04  prompts/steps/job-02-step-04-kineticsdb.md  KineticsDB facade + family-definition storage
  step 05  prompts/steps/job-02-step-05-miscdb.md  Transport/StatMech/Solvation facades + job-02 gate

## The job gate

Run by the final step's session (gates/gate_02.py, report
reports/job-02.md):

gates/gate_02.py: 1. Round-trip count: for each library in a fixed list (recommended_ libraries.yml 'default' set + primaryThermoLibrary + training depository): entry count via rmgdb == entry count via RMG-Py loading the same library (RMG-Py script -> gates/baselines/db_counts.json). Must be EXACT. 2. Content hash: primaryThermoLibrary + one kinetics library: dump all entries (label/sorted-smiles/coeffs/T-bounds) to sorted YAML from BOTH rmgdb (via rmgpu) and RMG-Py; diff byte-identical after normalization. 3. Lookup parity: 25 random species from superminimal/c3h4: thermo lookup via rmgpu == via RMG-Py (same model type + coefficients, tol 1e-12). 4. Rate-model round-trip: 100 reactions sampled from kinetics.db: build the rmgpu rate model from stored params; k(300,1bar) and k(1000,1bar) match RMG-Py's evaluation of the same params (rel tol 1e-10). 5. rmgdb coverage gaps documented (report + STATUS decisions if they hit later jobs).

## When the job is done

The final step's report (reports/job-02.md) has the gate result, the
job table row is `done` (or `blocked` with the cause), and the NEXT pointer
in STATUS.md targets job-03's first step (if there is a next job).
