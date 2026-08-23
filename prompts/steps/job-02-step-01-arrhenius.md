# job-02/step-01: Rate models: Arrhenius family + registry base

Job: job-02 - Database layer via rmgdb
Prereq: job-01 done (Molecule for substructure lookups)
This step is part of that job. The job's overall goal:
ALL database I/O for rmgpu goes through rmgdb - the SQL/SQLite wrapper over RMG-database. No YAML-parsing of RMG-database files inside rmgpu (except data rmgdb does not cover yet, which must be documented, not silently worked around). Deliverables: rmgpu/db/ (loaders + facades), rmgpu/data/ (entries, rate-model math, thermo models, kinetics retrieval), rmgpu/kinetics/models.py (the rate registry that later jobs consume).
The job's gate (run by the job's final step):
gates/gate_02.py: 1. Round-trip count: for each library in a fixed list (recommended_ libraries.yml 'default' set + primaryThermoLibrary + training depository): entry count via rmgdb == entry count via RMG-Py loading the same library (RMG-Py script -> gates/baselines/db_counts.json). Must be EXACT. 2. Content hash: primaryThermoLibrary + one kinetics library: dump all entries (label/sorted-smiles/coeffs/T-bounds) to sorted YAML from BOTH rmgdb (via rmgpu) and RMG-Py; diff byte-identical after normalization. 3. Lookup parity: 25 random species from superminimal/c3h4: thermo lookup via rmgpu == via RMG-Py (same model type + coefficients, tol 1e-12). 4. Rate-model round-trip: 100 reactions sampled from kinetics.db: build the rmgpu rate model from stored params; k(300,1bar) and k(1000,1bar) match RMG-Py's evaluation of the same params (rel tol 1e-10). 5. rmgdb coverage gaps documented (report + STATUS decisions if they hit later jobs).

First step of this job: skim /home/jackson/rmgpu/rmgpu/ORIENTATION.md once
(what stays/goes/external; the master-equation data path; conventions). Later
steps of this job do not need it - everything they need is in their own file.

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

- Previous step (already done, its code is in the tree): (this is the first step of the job)
- This step: job-02/step-01-arrhenius
- Next step (do NOT start it): job-02/step-02-falloff

## Goal

Port the MATH of RMG's kinetics rate models to numpy (NO Cython),
part 1: the Arrhenius family + the registry skeleton. This is the registry
every other job consumes - the API gets pinned here.
Read RMG-Py/rmgpy/data/thermo.py's GA/HBI code ONLY to confirm what is NOT
ported (deleted - PLAN.md 3): the port here is pure rate-expression math.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/kinetics/arrhenius.pyx  (2045 - Arrhenius,
    ArrheniusEP with Tmin/Tmax extrapolation, the forward/reverse rate
    machinery; port the math)
  RMG-Py/rmgpy/kinetics/kineticsdata.pyx  (271 - the Kinetics data class,
    storage semantics)
  RMG-Py/rmgpy/kinetics/model.pyx  (578 - the model registry:
    forward/reverse rate generation, generate_reverse_rate_coefficient via
    dG(T) integration - port the math; the registry shape is rmgpu's own)
  RMG-Py/rmgpy/data/thermo.py  (3099 - READ THE GA/HBI SECTIONS ONLY to
    confirm they are out of scope; do NOT port them)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/kinetics/models.py: the rate-expression registry
  (numpy/torch, thin, NO Cython):
    Arrhenius(A,n,Ea,T0 optional), ArrheniusEP (with Tmin/Tmax + the
    extrapolation RMG uses), storage classes for the rest (falloff step 3,
    Marcus step 3, Chebyshev/PDepArrhenius step 3 - the step files name the
    homes).
  - k(T), k(T,P) evaluation in SI; units per RMG's (A in its native unit,
    Ea in J/mol internally - document the convention at the class docstring).
  - forward/reverse: generate_reverse_rate_coefficient ported (thermodynamic
    consistency via dG(T) from the thermo models - the thermo models are
    step 4's; for now a protocol/duck-typed thermo interface + a numpy
    reference implementation in the test).
  - a thin RateModel union/registry: `make_rate_model(kind, **params)` and
    `.evaluate(T, P)`.
- tests/test_kinetics_models.py: k(T) on a grid vs RMG-Py's arrhenius.pyx
  evaluation (script via rmg_env; gate step 5 generalizes it); ArrheniusEP
  extrapolation behavior at T<Tmin/T>Tmax; reverse-rate consistency on a
  toy reaction (dG from a hand-computed thermo).

## Checks (must run and pass before you claim done)

  pytest tests/test_kinetics_models.py -q -> all pass
  reference: RMG-Py script evaluates the same params; max rel diff recorded

## Pitfalls

- Do NOT port the BM (Bayesian-modeling) estimator - deleted
  (PLAN.md 3); ArrheniusBM stays only as a STORAGE type (step 3 adds it).
- The registry API is load-bearing (jobs 04/06/07/09 consume it): keep it
  flat and explicit - dataclass-ish models, evaluate(T,P), no hidden state.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-02/step-01: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-02/step-01 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-02-step-01-arrhenius.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
