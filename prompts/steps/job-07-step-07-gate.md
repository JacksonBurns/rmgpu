# job-07/step-07: Job-07 gate (CSE k(T,P) parity, propane_branching)

Job: job-07 - Statmech + master equation (CSE) + pdep parity
Prereq: jobs 01-06 done (molecule, db, estimators, the core loop with its HPL pdep-stub, the reactor backend)
This step is part of that job. The job's overall goal:
Pressure dependence for unimolecular/reaction networks: statistical mechanics (DoS from vibrational/rotor/translation modes), the discretized master equation (CSE lumping - the default method), collision models, and the (T,P)-grid solving + Chebyshev/PDepArrhenius fitting that produces `Falloff` kinetics objects for the rate registry. NO QM anywhere (PLAN.md 8a): E0 from ML thermo, frequencies from the statmech DB, TS E0 derived from the HPL rate. The numerically hardest job in the project - CSE first, gated, before any other method (job 08).
The job's gate (run by the job's final step):
gates/gate_07.py (propane_branching, the R_Addition-heavy example, CSE in BOTH rmgpu and RMG-Py, same T/P grid): 1. Statmech unit parity: 50 species/intermediates from propane_branching: conformer assembly (E0, spin, mode counts) matches RMG-Py; Cp(T) on a 300-1500K grid within 1e-6 relative; DoS rho(E) on a grid within 1e-6 relative (spot-check 10, report max diff). 2. TS-E0 derivation: 20 pressure-dependent reactions: E0_TS (rmgpu, from the HPL rate) == E0_TS (RMG-Py) within 1e-8 (pins the no-QM path). 3. CSE k(T,P) parity (THE gate): for every pressure-dependent reaction: the fitted Falloff k(T,P) on the grid - max relative diff in k_inf, k_0, the broadening params + the Chebyshev/PDepArrhenius coefficients. TARGET: < 1% relative on k(T,P) values, < 1e-3 on coefficients. Any >1% reaction: listed + diagnosed (grain mismatch? collision? DoS?). 4. Network parity: network structure (isomers, channels, grain counts) matches RMG-Py (counts + a spot-check grain grid). 5. The full rmgpu run of propane_branching (pdep ON) completes; final core species/reaction counts vs RMG-Py (small divergence vs job 06's HPL run expected - the point is the k(T,P) parity above). A RED gate: port a second family's example (c3h4 has pdep reactions) to confirm it is not propane-specific.

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

- Previous step (already done, its code is in the tree): job-07/step-06-driver
- This step: job-07/step-07-gate
- Next step (do NOT start it): the job gate for job-07 (its step file)

## Goal

Run the job-07 gate: the statmech parity, the TS-E0 parity, the CSE
k(T,P) parity on propane_branching (THE gate), the network structure, the
full run. The numerically hardest gate - a RED is diagnosed to the
responsible sub-component (DoS? grains? collision? TS-E0?).

## Reference to read (this step's budget)

  (no new reference reads)
  /home/jackson/rmgpu/RMG-Py/examples/rmg/propane_branching/input.py (the
    example - its pressure_dependence block: the method + the T/P grid)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- gates/gate_07.py: implements the job definition (the five
  sub-gates: statmech parity (50 species), TS-E0 (20 reactions, 1e-8),
  CSE k(T,P) on propane_branching (every pdep reaction: the Falloff k(T,P)
  on the grid - max rel diff in k_inf, k_0, the broadening params + the
  Chebyshev/PDepArrhenius coefficients; target < 1% k(T,P), < 1e-3
  coefficients), the network structure (counts + a grain spot-check), the
  full run (core counts vs RMG-Py)).
- The RMG-Py reference: scripts/rmgpy_pdep_reference.py (env rmg_env) runs
  RMG-Py on propane_branching (CSE, the same T/P grid) + dumps the statmech
  (50 species: E0/spin/mode-counts/Cp(T)/DoS) + the TS-E0 (20 reactions) +
  the pdep reactions' fitted Falloff params + the k(T,P) grid + the
  network structures -> gates/baselines/propane_branching/ (commit).
- reports/job-07.md: all five sub-gates' numbers (real), any >1% reaction
  listed + diagnosed (grain mismatch? collision? DoS? TS-E0?), the wall-
  time (the GPU payoff), the CSE method string used, the blocked-families
  impact (if any pdep family was blocked in job 05).
- A RED k(T,P): the diagnosis protocol (from the job brief's pitfalls) -
  if k(T,P) disagrees but the DoS matches: look at grains/collision; if
  the DoS disagrees: the conformer assembly (step 3) or the modes (step
  1). Fix the responsible module + re-run. A RED that is not propane-
  specific (confirmed on c3h4's pdep reactions) is a BLOCKER - record it
  precisely.

## Checks (must run and pass before you claim done)

  /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_07.py
    -> the sub-gates' numbers recorded (target: < 1% k(T,P), < 1e-3
    coefficients, 1e-8 TS-E0, 1e-6 statmech)
  pytest tests/ -q -> all pass

## Pitfalls

- This is the numerically hardest gate (the plan says so) - a RED
  is expected to be diagnosed to the sub-component, not "fixed" by tuning
  a threshold. The sub-gates (statmech, TS-E0, toy-Lindemann) exist to
  isolate the cause - if a sub-gate passed but k(T,P) fails, the cause is
  in the composition (grains/collision/driver), not the parts.
- The reference is RMG-Py with CSE (the same method) - the method string
  must match (the long string 'chemically-significant eigenvalues'); a
  method mismatch (rmgpu CSE vs RMG-Py MSC) is a category error.
- The full run's core-count divergence vs job 06's HPL run is EXPECTED
  (pdep changes the rates) - the gate's parity target is the k(T,P) + the
  RMG-Py pdep-on comparison, not the HPL comparison.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-07/step-07: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-07/step-07 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-07-step-07-gate.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
