# job-12: Catalysis plugin (post-parity; the heavy plugin)

Status of this file: a JOB BRIEF, not a task. Do not try to "do this job"
in one session - it is decomposed into the steps below, one session each.
The human reads STATUS.md's NEXT pointer; this file is the map of the job.

## Goal

A rmgpu/plugins/catalysis/ package that makes surface-catalyzed mechanism generation work through the EXISTING core (the recipe engine, the ML estimators, the reactor interface) without forking any core module. The core sees only the plugin protocol. The big plugin: a second phase (the surface) with its own species type, families, thermo/kinetics providers, and a coverage-tracking reactor. Done only after gas+liquid parity (job 10/11).

## Prereq

job-11 done (the plugin protocol validated by solvation)

## Steps (strictly sequential; one fresh human-started session each)

  step 01  prompts/steps/job-12-step-01-species.md  SurfaceSpecies + the site graph
  step 02  prompts/steps/job-12-step-02-families.md  The Surface_* family recipes + the families loader
  step 03  prompts/steps/job-12-step-03-thermo-kinetics.md  The surface thermo + kinetics providers
  step 04  prompts/steps/job-12-step-04-reactor.md  The SurfaceReactor (the coverage state + the site balance)
  step 05  prompts/steps/job-12-step-05-gate.md  Job-12 gate (the catalysis plugin parity)

## The job gate

Run by the final step's session (gates/gate_12.py, report
reports/job-12.md):

gates/gate_12.py: 1. SurfaceSpecies round-trip: the site-graph construction + the adsorbate representation parity with RMG-Py (a fixed set of sites/adsorbates from the examples: the structure, the facet, the label). 2. Surface family application: 20 (site, adsorbate/gas-species) cases: the product enumeration (the job-05 engine + the plugin's families) matches RMG-Py (the products + the degeneracy). 3. Coverage-dependent kinetics: 15 surface reactions: k(coverage, T) from the plugin's rate models == RMG-Py's (surface.pyx), 1e-8. 4. Surface reactor: `minimal_surface` in rmgpu vs RMG-Py: the mechanism sets (the core species/reaction D) + the coverage profiles (the vacant-site fraction vs time) max diff. The site-balance constraint satisfied to the solver tolerance at every step (asserted in the gate - the DAE correctness check). 5. Plugin isolation: the catalysis code lives ONLY in rmgpu/plugins/catalysis/ + the registered hook call-sites; a gas- phase run with the plugin INSTALLED but no surface reactor is unchanged (the plugin is inert without a surface reactor).

## When the job is done

The final step's report (reports/job-12.md) has the gate result, the
job table row is `done` (or `blocked` with the cause), and the NEXT pointer
in STATUS.md targets job-13's first step (if there is a next job).
