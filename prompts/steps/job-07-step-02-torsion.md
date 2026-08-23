# job-07/step-02: Statmech torsions: 1D rotor PDE + 2D (ndTorsions)

Job: job-07 - Statmech + master equation (CSE) + pdep parity
Prereq: jobs 01-06 done (molecule, db, estimators, the core loop with its HPL pdep-stub, the reactor backend)
This step is part of that job. The job's overall goal:
Pressure dependence for unimolecular/reaction networks: statistical mechanics (DoS from vibrational/rotor/translation modes), the discretized master equation (CSE lumping - the default method), collision models, and the (T,P)-grid solving + Chebyshev/PDepArrhenius fitting that produces `Falloff` kinetics objects for the rate registry. NO QM anywhere (PLAN.md 8a): E0 from ML thermo, frequencies from the statmech DB, TS E0 derived from the HPL rate. The numerically hardest job in the project - CSE first, gated, before any other method (job 08).
The job's gate (run by the job's final step):
gates/gate_07.py (propane_branching, the R_Addition-heavy example, CSE in BOTH rmgpu and RMG-Py, same T/P grid): 1. Statmech unit parity: 50 species/intermediates from propane_branching: conformer assembly (E0, spin, mode counts) matches RMG-Py; Cp(T) on a 300-1500K grid within 1e-6 relative; DoS rho(E) on a grid within 1e-6 relative (spot-check 10, report max diff). 2. TS-E0 derivation: 20 pressure-dependent reactions: E0_TS (rmgpu, from the HPL rate) == E0_TS (RMG-Py) within 1e-8 (pins the no-QM path). 3. CSE k(T,P) parity (THE gate): for every pressure-dependent reaction: the fitted Falloff k(T,P) on the grid - max relative diff in k_inf, k_0, the broadening params + the Chebyshev/PDepArrhenius coefficients. TARGET: < 1% relative on k(T,P) values, < 1e-3 on coefficients. Any >1% reaction: listed + diagnosed (grain mismatch? collision? DoS?). 4. Network parity: network structure (isomers, channels, grain counts) matches RMG-Py (counts + a spot-check grain grid). 5. The full rmgpu run of propane_branching (pdep ON) completes; final core species/reaction counts vs RMG-Py (small divergence vs job 06's HPL run expected - the point is the k(T,P) parity above). A RED gate: port a second family's example (c3h4 has pdep reactions) to confirm it is not propane-specific.

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

- Previous step (already done, its code is in the tree): job-07/step-01-modes
- This step: job-07/step-02-torsion
- Next step (do NOT start it): job-07/step-03-assembly

## Goal

Port the torsion machinery: the 1D rotor (torsion.pyx) and the 2D
rotor (ndTorsions.py) - the hindered-rotor PDE solves that RMG uses for
torsional modes beyond the 1D approximation.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/statmech/torsion.pyx  (705 - the 1D rotor PDE (the
    torsional mode as a PDE in the torsional angle - port to scipy/torch)
  RMG-Py/rmgpy/statmech/ndTorsions.py  (740 - the 2D rotor (coupled
    torsions) - NOTE: RMG's runtime used arkane's ess_factory for the 2D
    scans - DELETED (PLAN.md 8a: no Arkane); the 2D TORSION MATH (the
    eigenproblem) is ported, the ESS submit machinery is NOT; the rotor
    barriers come from the statmech DB (group values) - see the data path)
  (the statmech/ modes from step 1)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/statmech/torsion.py: the 1D torsion rotor (the PDE form
  RMG uses - port to scipy/torch; the barrier + reduced mass as data;
  Cp(T) + the states per the PDE solution).
- rmgpu/statmech/ndtorsions.py: the 2D torsion (coupled) eigenproblem -
  the MATH only (the 2D Hamiltonian eigenproblem) - the ESS/Arkane
  machinery is NOT ported (deleted); the barriers that feed it come from
  the statmech DB (step 3's conformer assembly - the group values), per
  the master-equation data path (PLAN.md 8a.2).
- tests/test_torsions.py: the 1D rotor Cp(T) + states vs RMG-Py (the
  gate's 50-species set includes torsional species - the reference
  script); the 2D eigenproblem: energy levels vs RMG-Py's on a 2-
  coupled-torsion case (a small hand-built case; the ESS path is not
  exercised - document that boundary).

## Checks (must run and pass before you claim done)

  pytest tests/test_torsions.py -q -> all pass
  the 1D rotor: Cp(T) + states match RMG-Py (max rel diff recorded)

## Pitfalls

- The ESS boundary: RMG's 2D torsion SCAN used Arkane's ESS
  factory (deleted) - rmgpu's 2D torsion consumes barrier DATA (from the
  statmech DB) and solves the eigenproblem locally. Do NOT re-create the
  ESS submit machinery - it is out of scope (PLAN.md 7/8a).
- The 2D path is used rarely (a few species in the big examples) - if the
  eigenproblem is too heavy for a CPU solve, note it (the GPU is the
  payoff: torch.linalg on cuda) - but gate the 1D path first (step 6);
  the 2D is a bonus if the reference matches.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-07/step-02: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-07/step-02 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-07-step-02-torsion.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
