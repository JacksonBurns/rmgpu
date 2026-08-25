# job-12/step-03: The surface thermo + kinetics providers

Job: job-12 - Catalysis plugin (post-parity; the heavy plugin)
Prereq: job-11 done (the plugin protocol validated by solvation)
This step is part of that job. The job's overall goal:
A rmgpu/plugins/catalysis/ package that makes surface-catalyzed mechanism generation work through the EXISTING core (the recipe engine, the ML estimators, the reactor interface) without forking any core module. The core sees only the plugin protocol. The big plugin: a second phase (the surface) with its own species type, families, thermo/kinetics providers, and a coverage-tracking reactor. Done only after gas+liquid parity (job 10/11).
The job's gate (run by the job's final step):
gates/gate_12.py: 1. SurfaceSpecies round-trip: the site-graph construction + the adsorbate representation parity with RMG-Py (a fixed set of sites/adsorbates from the examples: the structure, the facet, the label). 2. Surface family application: 20 (site, adsorbate/gas-species) cases: the product enumeration (the job-05 engine + the plugin's families) matches RMG-Py (the products + the degeneracy). 3. Coverage-dependent kinetics: 15 surface reactions: k(coverage, T) from the plugin's rate models == RMG-Py's (surface.pyx), 1e-8. 4. Surface reactor: `minimal_surface` in rmgpu vs RMG-Py: the mechanism sets (the core species/reaction D) + the coverage profiles (the vacant-site fraction vs time) max diff. The site-balance constraint satisfied to the solver tolerance at every step (asserted in the gate - the DAE correctness check). 5. Plugin isolation: the catalysis code lives ONLY in rmgpu/plugins/catalysis/ + the registered hook call-sites; a gas- phase run with the plugin INSTALLED but no surface reactor is unchanged (the plugin is inert without a surface reactor).

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

- Previous step (already done, its code is in the tree): job-12/step-02-families
- This step: job-12/step-03-thermo-kinetics
- Next step (do NOT start it): job-12/step-04-reactor

## Goal

The coverage-dependent thermo + kinetics providers: the adsorption-
energy thermo (the library / the ML model + the linear-scaling correction)
+ the surface rate models (the coverage-dependent pre-exponential, the
site-balance terms, the BEP relations) - ported onto the job-02/04 rate
registry (new model types; the existing registry stays).

## Reference to read (this step's budget)

  RMG-Py/rmgpy/kinetics/surface.pyx  (1291 - the surface rate
    expressions: the coverage-dependent pre-exponential, the site-balance
    terms, the BEP relations - port the MATH onto the job-02/04 registry
    (new model types: SurfaceArrhenius / SurfaceArrheniusBEP - the
    coverage-theta terms))
  RMG-Py/rmgpy/data/thermo.py  (the correct_binding_energy / the
    set_binding_energies - the linear-scaling correction of the
    adsorption energies against a reference metal - the coverage-
    dependent thermo provider)
  (step 12/1's SurfaceSpecies - the adsorption energy it carries; the
    job-04's estimators - the ML adsorption model is just another Chemprop
    species model (PLAN.md 9.2); the job-02/04's registry)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/plugins/catalysis/thermo.py:
    `SurfaceThermoProvider` (implements register_thermo): for a surface
    species, the adsorption energy (from the adsorbate library / the ML
    adsorption model) + the optional coverage-dependent + the linear-
    scaling correction (the correct_binding_energy ported - the linear
    scaling against a reference metal) - a thermo PROVIDER (not a new
    engine - the core composes it with the gas-phase thermo (step 12/1's
    adsorbate)).
- rmgpu/plugins/catalysis/kinetics.py:
    `SurfaceKineticsProvider` (implements register_kinetics): the surface
    rate-model providers (the coverage-dependent, the BEP) - the new model
    types (SurfaceArrhenius / SurfaceArrheniusBEP) added to the job-02/04
    registry (the registry gets them - the existing types stay; the
    surface models carry the extra coverage-theta terms).
    the k(coverage, T) evaluation (the theta dependence - the coverage is
    the reactor's state (step 12/4) - the model takes it as an argument).
- tests/test_surface_kinetics.py: 15 surface reactions: k(coverage, T)
  from the plugin's rate models vs RMG-Py (surface.pyx) - 1e-8 (the gate's
  check 3 is the full parity; this step: a subset + the model-type
  round-trip (the params -> the model -> the params)).
  the linear-scaling correction: the corrected adsorption energy vs
  RMG-Py (the correct_binding_energy - a subset).

## Checks (must run and pass before you claim done)

  pytest tests/test_surface_kinetics.py -q -> all pass
  the 15 reactions: k(coverage, T) vs RMG-Py (1e-8)

## Pitfalls

- The registry is EXTENDED (the new model types) - not forked (the
  core's registry stays; the surface types are new entries - the
  evaluate(T, P) signature gains the coverage argument for the surface
  types (a protocol note - the registry is the job-04 API, an extension
  is a documented change, note it).
- The coverage-theta terms couple the thermo + the kinetics (the
  adsorption energy shifts with the coverage; the rates carry the theta)
  - keep the providers SEPARATE (the thermo provider returns the
  corrected energy; the kinetics provider carries the theta terms) - the
  core composes them (PLAN.md 9.2).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-12/step-03: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-12/step-03 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-12-step-03-thermo-kinetics.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
