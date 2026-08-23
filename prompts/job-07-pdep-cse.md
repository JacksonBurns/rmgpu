# job-07: Statmech + master equation (CSE) + pdep parity

Status of this file: a JOB BRIEF, not a task. Do not try to "do this job"
in one session - it is decomposed into the steps below, one session each.
The coordinator (see README.md, "Session vs step") picks the next step from
STATUS.md's NEXT pointer; this file is the map of the job.

## Goal

Pressure dependence for unimolecular/reaction networks: statistical mechanics (DoS from vibrational/rotor/translation modes), the discretized master equation (CSE lumping - the default method), collision models, and the (T,P)-grid solving + Chebyshev/PDepArrhenius fitting that produces `Falloff` kinetics objects for the rate registry. NO QM anywhere (PLAN.md 8a): E0 from ML thermo, frequencies from the statmech DB, TS E0 derived from the HPL rate. The numerically hardest job in the project - CSE first, gated, before any other method (job 08).

## Prereq

jobs 01-06 done (molecule, db, estimators, the core loop with its HPL pdep-stub, the reactor backend)

## Steps (strictly sequential; one fresh subagent session each)

  step 01  prompts/steps/job-07-step-01-modes.md  Statmech modes: conformer, vibration, rotation
  step 02  prompts/steps/job-07-step-02-torsion.md  Statmech torsions: 1D rotor PDE + 2D (ndTorsions)
  step 03  prompts/steps/job-07-step-03-assembly.md  Conformer assembly from the statmech DB (no QM)
  step 04  prompts/steps/job-07-step-04-network.md  The pdep network + master equation (CSE) + TS-E0
  step 05  prompts/steps/job-07-step-05-collision.md  Collision models: CSE + collision frequency
  step 06  prompts/steps/job-07-step-06-driver.md  The pdep driver + loop wiring + pdep/ output
  step 07  prompts/steps/job-07-step-07-gate.md  Job-07 gate (CSE k(T,P) parity, propane_branching)

## The job gate

Run by the final step's session (gates/gate_07.py, report
reports/job-07.md):

gates/gate_07.py (propane_branching, the R_Addition-heavy example, CSE in BOTH rmgpu and RMG-Py, same T/P grid): 1. Statmech unit parity: 50 species/intermediates from propane_branching: conformer assembly (E0, spin, mode counts) matches RMG-Py; Cp(T) on a 300-1500K grid within 1e-6 relative; DoS rho(E) on a grid within 1e-6 relative (spot-check 10, report max diff). 2. TS-E0 derivation: 20 pressure-dependent reactions: E0_TS (rmgpu, from the HPL rate) == E0_TS (RMG-Py) within 1e-8 (pins the no-QM path). 3. CSE k(T,P) parity (THE gate): for every pressure-dependent reaction: the fitted Falloff k(T,P) on the grid - max relative diff in k_inf, k_0, the broadening params + the Chebyshev/PDepArrhenius coefficients. TARGET: < 1% relative on k(T,P) values, < 1e-3 on coefficients. Any >1% reaction: listed + diagnosed (grain mismatch? collision? DoS?). 4. Network parity: network structure (isomers, channels, grain counts) matches RMG-Py (counts + a spot-check grain grid). 5. The full rmgpu run of propane_branching (pdep ON) completes; final core species/reaction counts vs RMG-Py (small divergence vs job 06's HPL run expected - the point is the k(T,P) parity above). A RED gate: port a second family's example (c3h4 has pdep reactions) to confirm it is not propane-specific.

## When the job is done

The final step's report (reports/job-07.md) has the gate result, the
job table row is `done` (or `blocked` with the cause), and the NEXT pointer
in STATUS.md targets job-08's first step (if there is a next job).
