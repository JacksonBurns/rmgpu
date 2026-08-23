# job-12/step-02: The Surface_* family recipes + the families loader

Job: job-12 - Catalysis plugin (post-parity; the heavy plugin)
Prereq: job-11 done (the plugin protocol validated by solvation)
This step is part of that job. The job's overall goal:
A rmgpu/plugins/catalysis/ package that makes surface-catalyzed mechanism generation work through the EXISTING core (the recipe engine, the ML estimators, the reactor interface) without forking any core module. The core sees only the plugin protocol. The big plugin: a second phase (the surface) with its own species type, families, thermo/kinetics providers, and a coverage-tracking reactor. Done only after gas+liquid parity (job 10/11).
The job's gate (run by the job's final step):
gates/gate_12.py: 1. SurfaceSpecies round-trip: the site-graph construction + the adsorbate representation parity with RMG-Py (a fixed set of sites/adsorbates from the examples: the structure, the facet, the label). 2. Surface family application: 20 (site, adsorbate/gas-species) cases: the product enumeration (the job-05 engine + the plugin's families) matches RMG-Py (the products + the degeneracy). 3. Coverage-dependent kinetics: 15 surface reactions: k(coverage, T) from the plugin's rate models == RMG-Py's (surface.pyx), 1e-8. 4. Surface reactor: `minimal_surface` in rmgpu vs RMG-Py: the mechanism sets (the core species/reaction D) + the coverage profiles (the vacant-site fraction vs time) max diff. The site-balance constraint satisfied to the solver tolerance at every step (asserted in the gate - the DAE correctness check). 5. Plugin isolation: the catalysis code lives ONLY in rmgpu/plugins/catalysis/ + the registered hook call-sites; a gas- phase run with the plugin INSTALLED but no surface reactor is unchanged (the plugin is inert without a surface reactor).

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

- Previous step (already done, its code is in the tree): job-12/step-01-species
- This step: job-12/step-02-families
- Next step (do NOT start it): job-12/step-03-thermo-kinetics

## Goal

The ~60 Surface_* families: load their recipes from the DB (they are
recipe DATA - the same DSL as the core families - they flow through the
job-05 recipe engine). The plugin supplies the site templates + the
surface-reactant patterns; NO new generator.

## Reference to read (this step's budget)

  RMG-database/input/kinetics/families/Surface_*/  (the ~60 surface
    families: the recipes - the adsorption single/double/vdW/dissociative,
    the dissociation, the migration, the abstraction (the Eley-Rideal /
    the Langmuir-Hinshelwood), the beta-scission, the proton/electron
    transfer - the recipe DATA; inspect 3-4 (the Surface_Adsorption_*,
    one abstraction) to pin the format)
  (job-05's recipe engine + family loader - the families flow through it;
    the job-05 step-4's family-storage decision applies)
  (step 12/1's SurfaceSpecies - the surface-reactant patterns)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/plugins/catalysis/families.py:
    `load_surface_families()` - load the Surface_* family recipes from the
    DB (the job-02/05 machinery - they are recipes; the plugin supplies
    the site templates + the surface-reactant patterns - the patterns
    that match a SurfaceSpecies (the adsorbate + the site) + a gas
    species).
    NO new generator - the job-05 recipe engine (ReactionRecipe,
    generate_reactions) runs them (the surface families are the same DSL
    - the atom-labeled bond-ops over the site-graph + the gas molecule).
    the KineticsFamilies extension: the surface families join the
    families list (the core's match_reaction finds them for a surface
    reaction).
- tests/test_surface_families.py: 20 (site, adsorbate/gas-species) cases:
  the product enumeration (the job-05 engine + the plugin's families) vs
  RMG-Py (the products + the degeneracy - the gate's check 2 is the full
  parity; this step: a subset + the format check).
  the blocked-surface-families list (any family whose recipe uses a
  construct the site-graph Molecule does not support - each documented).

## Checks (must run and pass before you claim done)

  pytest tests/test_surface_families.py -q -> all pass
  the 20 cases: the product enumeration (the products + the degeneracy)
    vs RMG-Py (the subset parity)

## Pitfalls

- The surface families are recipe DATA - a family that "needs a new
  generator" is a mis-port (the recipe engine is the generator); if a
  construct is missing, it is the site-graph Molecule (step 1) that needs
  it (add the minimum, note it) - not a new engine.
- The blocked-surface-families list is load-bearing (the gate proceeds
  around them) - an honest list, each with the exact construct.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-12/step-02: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-12/step-02 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-12-step-02-families.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
