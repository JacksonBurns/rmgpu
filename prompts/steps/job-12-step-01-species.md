# job-12/step-01: SurfaceSpecies + the site graph

Job: job-12 - Catalysis plugin (post-parity; the heavy plugin)
Prereq: job-11 done (the plugin protocol validated by solvation)
This step is part of that job. The job's overall goal:
A rmgpu/plugins/catalysis/ package that makes surface-catalyzed mechanism generation work through the EXISTING core (the recipe engine, the ML estimators, the reactor interface) without forking any core module. The core sees only the plugin protocol. The big plugin: a second phase (the surface) with its own species type, families, thermo/kinetics providers, and a coverage-tracking reactor. Done only after gas+liquid parity (job 10/11).
The job's gate (run by the job's final step):
gates/gate_12.py: 1. SurfaceSpecies round-trip: the site-graph construction + the adsorbate representation parity with RMG-Py (a fixed set of sites/adsorbates from the examples: the structure, the facet, the label). 2. Surface family application: 20 (site, adsorbate/gas-species) cases: the product enumeration (the job-05 engine + the plugin's families) matches RMG-Py (the products + the degeneracy). 3. Coverage-dependent kinetics: 15 surface reactions: k(coverage, T) from the plugin's rate models == RMG-Py's (surface.pyx), 1e-8. 4. Surface reactor: `minimal_surface` in rmgpu vs RMG-Py: the mechanism sets (the core species/reaction D) + the coverage profiles (the vacant-site fraction vs time) max diff. The site-balance constraint satisfied to the solver tolerance at every step (asserted in the gate - the DAE correctness check). 5. Plugin isolation: the catalysis code lives ONLY in rmgpu/plugins/catalysis/ + the registered hook call-sites; a gas- phase run with the plugin INSTALLED but no surface reactor is unchanged (the plugin is inert without a surface reactor).

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
- This step: job-12/step-01-species
- Next step (do NOT start it): job-12/step-02-families

## Goal

The surface species: SurfaceSpecies = the adsorbate + the site. The
site is a small site-graph (a surface atom + the neighbor atoms, the facet
info). Port RMG's representation - registered via register_species so the
core loop holds it like any species.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/species.py  (1177 - the Species + the SurfaceSpecies
    (the adsorbate + the site): the site-graph representation (a surface
    atom + the neighbors, the facet), the label, the charge/spin - port
    the representation)
  RMG-database/input/surface/ (or input/surface/libraries - locate it: the
    adsorbate library (the adsorption energies), the surface site
    definitions - the data the species consumes)
  RMG-Py/examples/rmg/minimal_surface/input.py (the surface example - the
    sites + the adsorbates it uses - the round-trip set)
  (job-01's Molecule - the site graph is a small Molecule (the metal atom
    + the neighbors))

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/plugins/catalysis/species.py:
    `SurfaceSpecies` (the adsorbate Molecule + the site-graph (a small
    Molecule: the surface atom + the neighbors, the facet info), the
    label, the charge/spin) - the RMG representation ported.
    registered via register_species (the plugin's registration - the core
    loop holds it in core/edge like any species - the core's species
    handling must accept it (a note if the core needs a minimal hook -
    the core stays agnostic, the species is a data object)).
    the adsorbate representation: the adsorption energy (from the
    adsorbate library / the ML adsorption model - the data from the
    surface DB) - the species carries it (the thermo provider, step 3,
    uses it).
- tests/test_surface_species.py: a fixed set of sites/adsorbates (from the
  minimal_surface example): the site-graph construction + the adsorbate
  representation (the structure, the facet, the label) vs RMG-Py (a
  reference script - the gate generalizes it).

## Checks (must run and pass before you claim done)

  pytest tests/test_surface_species.py -q -> all pass
  the site/adsorbate set: the representation vs RMG-Py (the structure,
    the facet, the label match)

## Pitfalls

- The site graph is a small Molecule (the metal + the neighbors) -
  reuse job-01's Molecule, do NOT invent a new graph type (the core's
  recipe engine operates on Molecules; the surface families (step 2)
  need the site as a Molecule).
- The adsorption energy is DATA (the library / the ML model) - the species
  carries it; the thermo provider (step 3) interprets it - do NOT compute
  it in the species (that is the provider's job).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-12/step-01: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-12/step-01 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-12-step-01-species.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
