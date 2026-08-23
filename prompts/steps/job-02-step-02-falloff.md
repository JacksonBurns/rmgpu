# job-02/step-02: Rate models: falloff, Chebyshev, Marcus, tunneling

Job: job-02 - Database layer via rmgdb
Prereq: job-01 done (Molecule for substructure lookups)
This step is part of that job. The job's overall goal:
ALL database I/O for rmgpu goes through rmgdb - the SQL/SQLite wrapper over RMG-database. No YAML-parsing of RMG-database files inside rmgpu (except data rmgdb does not cover yet, which must be documented, not silently worked around). Deliverables: rmgpu/db/ (loaders + facades), rmgpu/data/ (entries, rate-model math, thermo models, kinetics retrieval), rmgpu/kinetics/models.py (the rate registry that later jobs consume).
The job's gate (run by the job's final step):
gates/gate_02.py: 1. Round-trip count: for each library in a fixed list (recommended_ libraries.yml 'default' set + primaryThermoLibrary + training depository): entry count via rmgdb == entry count via RMG-Py loading the same library (RMG-Py script -> gates/baselines/db_counts.json). Must be EXACT. 2. Content hash: primaryThermoLibrary + one kinetics library: dump all entries (label/sorted-smiles/coeffs/T-bounds) to sorted YAML from BOTH rmgdb (via rmgpu) and RMG-Py; diff byte-identical after normalization. 3. Lookup parity: 25 random species from superminimal/c3h4: thermo lookup via rmgpu == via RMG-Py (same model type + coefficients, tol 1e-12). 4. Rate-model round-trip: 100 reactions sampled from kinetics.db: build the rmgpu rate model from stored params; k(300,1bar) and k(1000,1bar) match RMG-Py's evaluation of the same params (rel tol 1e-10). 5. rmgdb coverage gaps documented (report + STATUS decisions if they hit later jobs).

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

- Previous step (already done, its code is in the tree): job-02/step-01-arrhenius
- This step: job-02/step-02-falloff
- Next step (do NOT start it): job-02/step-03-thermodb

## Goal

Port the MATH of the remaining rate models (part 2): falloff
kinetics (Lindemann/Troe/ThirdBody + efficient bath gases), the pdep
interpolation models (Chebyshev, PDepArrhenius), Marcus, tunneling (Wigner
at minimum), ArrheniusBM as storage.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/kinetics/falloff.pyx  (429 - Lindemann/Troe/ThirdBody
    + efficient-bath-gas handling; the k(T,P) wrapper + Troe broadening
    params)
  RMG-Py/rmgpy/kinetics/chebyshev.pyx  (299)
  RMG-Py/rmgpy/kinetics/marcus.pyx (if present; else grep -l marcus RMG-Py/
    rmgpy/kinetics/)
  RMG-Py/rmgpy/kinetics/tunneling.pyx  (265 - Wigner at minimum; keep Eckart
    only if RMG-Py exposes it publicly - check and note the decision)
  RMG-Py/rmgpy/data/kinetics/kineticsdata.pyx (Marcus/PDepArrhenius storage
    semantics if not covered by kineticsdata.pyx)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/kinetics/models.py (extend step 2):
    Falloff (k0, kinf model, Troe/Lindemann broadening, ThirdBody with
    efficient bath gases), Chebyshev (coefficient grid + the (T,P) eval
    RMG uses), PDepArrhenius (storage + eval), Marcus, ArrheniusBM (STORAGE
    ONLY - the BM estimator is deleted), tunneling (Wigner factor; Eckart if
    exposed).
  - k(T,P) evaluation in SI; the registry's evaluate(T,P) dispatches across
    all models.
- tests: k(T,P) on grids vs RMG-Py for each model (gate step 5 generalizes);
  Troe broadening on a real Troe entry from kinetics.db (find one: query
  the db); Marcus on a stored Marcus entry; Wigner factor on a stored value.

## Checks (must run and pass before you claim done)

  pytest tests/test_kinetics_models.py -q -> all pass

## Pitfalls

- PDepArrhenius/Chebyshev are what job 07 PRODUCES - the eval must
  be exact, the fit is job 07's.
- Efficient bath gas handling (sum of efficiencies x partial pressures) is
  fussy - port RMG's exact combination, test against a multi-bath-gas
  entry.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-02/step-02: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-02/step-02 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-02-step-02-falloff.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
