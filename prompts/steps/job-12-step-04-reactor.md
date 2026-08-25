# job-12/step-04: The SurfaceReactor (the coverage state + the site balance)

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

- Previous step (already done, its code is in the tree): job-12/step-03-thermo-kinetics
- This step: job-12/step-04-reactor
- Next step (do NOT start it): job-12/step-05-gate

## Goal

The SurfaceReactor: tracks the site coverage (the vacant vs the
occupied sites) as the extra state + the site-balance ALGEBRAIC constraint
(the DAE index-1 case torchdae handles). Ported onto the torchdae
backend. Registered via register_reactor.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/solver/surface.pyx  (1110 - the SurfaceReactor: the
    site coverage (the vacant vs the occupied) as the extra state + the
    site-balance ALGEBRAIC constraint (the DAE index-1 - torchdae's index
    reduction handles it) - port the MATH onto the torchdae backend)
  (job-06/1's torch backend - the DAE; the step 12/3's surface kinetics
    (the theta terms - the coverage is the state the kinetics reads))
  torchdae (the DAE + the index reduction - confirm the installed version
    handles the algebraic constraint; if it struggles, that is a
    torchdae finding (PLAN.md 13 risk 3), recorded, NOT a solver fork)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/plugins/catalysis/reactor.py:
    `SurfaceReactor` (port surface.pyx onto the torchdae backend): the
    state = the gas/liquid mole fractions + the site coverage (the
    fractions of the vacant vs the occupied sites - the per-site-type
    coverage for the multi-surface); the ODE/DAE rhs = the surface
    reaction rates (the step 12/3's models, the theta terms from the
    coverage state) + the site-balance ALGEBRAIC constraint (the sum of
    the site coverages = 1 per site type - the algebraic part, the DAE
    index-1).
    the site-balance constraint is ENFORCED by torchdae's index reduction
    (the algebraic constraint) - the gate asserts it is satisfied to the
    solver tolerance at every step (the DAE correctness check).
    registered via register_reactor (the core's reactor factory gets it).
- The multi-surface (minimal_multisurf): several sites at once - the state
  is the per-site-type coverage; the hard case (the gate gates the single-
  surface first, then the multi-site).
- tests/test_surface_reactor.py: a surface toy system (a 1-site, 1-
  adsorbate, 1-reaction): the coverage profile (the vacant-site fraction
  vs time) vs RMG-Py (the max diff); the site-balance constraint
  (the sum of the coverages = 1) asserted at every step (the DAE
  correctness); the multi-surface (2 sites): the per-site coverage vs
  RMG-Py (a subset).

## Checks (must run and pass before you claim done)

  pytest tests/test_surface_reactor.py -q -> all pass
  the surface toy: the coverage profile vs RMG-Py (the max diff recorded)
    + the site-balance constraint satisfied (asserted)

## Pitfalls

- The site-balance constraint is the DAE's algebraic part -
  torchdae's index reduction must handle it (PLAN.md 9.2 calls it out);
  if torchdae struggles, that is a torchdae FINDING (PLAN.md 13 risk 3) -
  record it, do NOT fork the solver (the no-second-backend anti-goal,
  PLAN.md 14).
- The coverage is the STATE (the reactor owns it) - the kinetics (step
  12/3) READS it (the theta terms) - do NOT duplicate the coverage in the
  kinetics (the single source of truth is the reactor's state).
- The multi-surface is the hard case - gate the single-surface first (this
  step), the multi-site is the gate's (step 12/5's check 4 extends it).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-12/step-04: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-12/step-04 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-12-step-04-reactor.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
