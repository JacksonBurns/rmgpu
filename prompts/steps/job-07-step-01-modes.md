# job-07/step-01: Statmech modes: conformer, vibration, rotation

Job: job-07 - Statmech + master equation (CSE) + pdep parity
Prereq: jobs 01-06 done (molecule, db, estimators, the core loop with its HPL pdep-stub, the reactor backend)
This step is part of that job. The job's overall goal:
Pressure dependence for unimolecular/reaction networks: statistical mechanics (DoS from vibrational/rotor/translation modes), the discretized master equation (CSE lumping - the default method), collision models, and the (T,P)-grid solving + Chebyshev/PDepArrhenius fitting that produces `Falloff` kinetics objects for the rate registry. NO QM anywhere (PLAN.md 8a): E0 from ML thermo, frequencies from the statmech DB, TS E0 derived from the HPL rate. The numerically hardest job in the project - CSE first, gated, before any other method (job 08).
The job's gate (run by the job's final step):
gates/gate_07.py (propane_branching, the R_Addition-heavy example, CSE in BOTH rmgpu and RMG-Py, same T/P grid): 1. Statmech unit parity: 50 species/intermediates from propane_branching: conformer assembly (E0, spin, mode counts) matches RMG-Py; Cp(T) on a 300-1500K grid within 1e-6 relative; DoS rho(E) on a grid within 1e-6 relative (spot-check 10, report max diff). 2. TS-E0 derivation: 20 pressure-dependent reactions: E0_TS (rmgpu, from the HPL rate) == E0_TS (RMG-Py) within 1e-8 (pins the no-QM path). 3. CSE k(T,P) parity (THE gate): for every pressure-dependent reaction: the fitted Falloff k(T,P) on the grid - max relative diff in k_inf, k_0, the broadening params + the Chebyshev/PDepArrhenius coefficients. TARGET: < 1% relative on k(T,P) values, < 1e-3 on coefficients. Any >1% reaction: listed + diagnosed (grain mismatch? collision? DoS?). 4. Network parity: network structure (isomers, channels, grain counts) matches RMG-Py (counts + a spot-check grain grid). 5. The full rmgpu run of propane_branching (pdep ON) completes; final core species/reaction counts vs RMG-Py (small divergence vs job 06's HPL run expected - the point is the k(T,P) parity above). A RED gate: port a second family's example (c3h4 has pdep reactions) to confirm it is not propane-specific.

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
- This step: job-07/step-01-modes
- Next step (do NOT start it): job-07/step-02-torsion

## Goal

Port the statmech MODES: the Conformer container + the vibrational
and rotational modes (harmonic oscillators, linear/nonlinear rotors,
hindered + free rotors, translation). The eigenproblems (hindered rotors,
torsions) go to scipy.linalg / torch.linalg - document the choice.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/statmech/conformer.pyx  (622 - the Conformer: E0,
    spin multiplicity, the modes list, get_heat_capacity, the
    get_number_of_states (DoS) entry points)
  RMG-Py/rmgpy/statmech/mode.pyx  (126 - the Mode base)
  RMG-Py/rmgpy/statmech/vibration.pyx  (255 - HarmonicOscillator: the
    heat capacity + the vibrational states)
  RMG-Py/rmgpy/statmech/rotation.pyx  (804 - LinearRotor, NonlinearRotor,
    the moment-of-inertia handling; HinderedRotor + FreeRotor (1D) - the
    hindered-rotor eigenproblem)
  RMG-Py/rmgpy/statmech/translation.pyx  (205 - Translation)
  RMG-Py/rmgpy/statmech/schrodinger.pyx  (296 - the 1D rotor
    eigenproblem solver - port to scipy.linalg/torch.linalg; the choice
    + why in the report)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/statmech/modes.py (or a statmech/ package per module):
    Mode base, HarmonicOscillator, LinearRotor, NonlinearRotor,
    HinderedRotor, FreeRotor, Translation - each: get_heat_capacity(T),
    get_number_of_states(E) (the state count up to energy E), the
    characteristic quantities (frequency, moments of inertia, barrier,
    reduced mass) as data.
    Conformer: E0 (J/mol), spin multiplicity, the modes list,
    get_heat_capacity(T) (sum over modes + translation),
    get_number_of_states(E) (the convolution/recursion over modes - the
    DoS core; port RMG's algorithm - this is hot, vectorize with numpy,
    correctness first), get_zero_point_energy if RMG carries it.
    Symmetry factors (job-01 symmetry.py) wired into the DoS (the
    symmetry number divides the rotational states - per RMG).
- tests/test_statmech_modes.py: per mode type, Cp(T) on a grid +
  get_number_of_states(E) on a grid vs RMG-Py (a reference script in env
  rmg_env - the gate generalizes it): max rel diff recorded (target 1e-6
  per the gate); the hindered-rotor eigenproblem: the energy levels vs
  RMG-Py's (the schrodinger port - a small basis-set comparison).

## Checks (must run and pass before you claim done)

  pytest tests/test_statmech_modes.py -q -> all pass
  reference: RMG-Py script dumps Cp(T) + state counts for a set of
    conformers; max rel diff < 1e-6 (or the deviation documented)

## Pitfalls

- The DoS convolution/recursion is the numerical core - a subtle
  bug (symmetry factor, ZPE, the grain boundary handling) silently shifts
  k(T,P). The gate's DoS sub-gate (step 6) catches it before the k(T,P)
  comparison - a deviation here is a BLOCKER for the k(T,P) gate.
- 1D/2D rotor eigenproblems: scipy.linalg (eigh on the discretized
  Hamiltonian) or torch.linalg - document the choice; the accuracy must
  match RMG-Py's (its own solver) to the gate's tolerance.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-07/step-01: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-07/step-01 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-07-step-01-modes.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
