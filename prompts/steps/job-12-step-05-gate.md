# job-12/step-05: Job-12 gate (the catalysis plugin parity)

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

- Previous step (already done, its code is in the tree): job-12/step-04-reactor
- This step: job-12/step-05-gate
- Next step (do NOT start it): the job gate for job-12 (its step file)

## Goal

Run the job-12 gate: the SurfaceSpecies round-trip, the surface
family application (20 cases), the coverage-dependent kinetics (15
reactions), the surface reactor (minimal_surface + the site balance), the
plugin isolation. The design-validated implementation of the heavy plugin.

## Reference to read (this step's budget)

  (no new reference reads)
  RMG-Py/examples/rmg/{minimal_surface,minimal_multisurf}/input.py +
    test/regression/{minimal_surface,RMS_liquidSurface_ch4o2cat}/ (the
    surface examples - the parity set)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- gates/gate_12.py: the five checks (the SurfaceSpecies round-
  trip: the site-graph + the adsorbate representation vs RMG-Py (the
  fixed set); the surface family application: 20 cases (the products +
  the degeneracy) vs RMG-Py; the coverage-dependent kinetics: 15
  reactions (k(coverage, T) 1e-8); the surface reactor: minimal_surface
  (the mechanism sets + the coverage profiles (the vacant-site fraction
  vs time) max diff + the site-balance constraint asserted to the solver
  tolerance at every step) + the minimal_multisurf (the multi-site, if
  step 12/4's multi-surface is done); the plugin isolation: the catalysis
  code ONLY in rmgpu/plugins/catalysis/ + the hook call-sites; a gas-
  phase run with the plugin INSTALLED but no surface reactor is
  unchanged).
- The RMG-Py references: the surface set (the species, the family
  products, the kinetics, the reactor profiles) ->
  gates/baselines/job12/ (commit).
- reports/job-12.md: the implementation mapped to PLAN.md 9.2 (each
  bullet of 9.2 -> where it lives in the plugin), + any protocol change
  forced by reality (if any, PLAN.md 9.1 updated), + the torchdae DAE
  status (the site-balance constraint - handled / the finding).
- The full tests/ suite green.

## Checks (must run and pass before you claim done)

  /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_12.py
    -> the five checks recorded (targets: the species round-trip, the
    20-case family parity, the 15-reaction kinetics 1e-8, the minimal_
    surface mechanism + coverage parity + the site balance, the isolation
    clean)
  pytest tests/ -q -> all pass

## Pitfalls

- The site-balance constraint (check 4) is the DAE correctness
  check - a constraint violation is a torchdae finding (PLAN.md 13 risk
  3) OR a port bug (the constraint not wired) - diagnose which (the toy
  from step 12/4 isolates it).
- The plugin isolation (check 5) is the design invariant (the plugin
  reuses the core, not a fork - PLAN.md 9.2) - a catalysis symbol in
  rmgpu/core is a violation; refactor it into the plugin.
- The multi-surface (minimal_multisurf) is the hard case - if the single-
  surface is green but the multi-surface is not, record it (the job is
  the single-surface parity; the multi-surface is a documented extension,
  not a blocker - note it in the report + the STATUS).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-12/step-05: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-12/step-05 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-12-step-05-gate.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
