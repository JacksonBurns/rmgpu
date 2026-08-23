# job-07/step-05: Collision models: CSE + collision frequency

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

- Previous step (already done, its code is in the tree): job-07/step-04-network
- This step: job-07/step-05-collision
- Next step (do NOT start it): job-07/step-06-driver

## Goal

The collision side: the CSE collision model (the default method -
port FIRST per the plan), the collision frequency (LJ params from the
transport DB), and the grain configuration logic. CSE plugs into step 4's
network via the protocol.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/pdep/cse.pyx  (385 - the CSE collision model (the
    default): the chemically-significant-eigenvalue lumping - port it; the
    allen + georgievskii variants (the method strings from step 4))
  RMG-Py/rmgpy/pdep/collision.pyx  (315 - the collision model base + the
    collision-energy-transfer handling)
  RMG-Py/rmgpy/pdep/configuration.pyx  (492 - the grain configuration +
    the collision frequency (calculate_collision_frequency))
  (the network protocol from step 4; job-02's TransportDB for the LJ
    params)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/pdep/collision.py:
    the collision-frequency calculation (calculate_collision_frequency -
    from the LJ sigma/epsilon via the transport DB (job 02) + the bath
    gas composition - port RMG's formula), the grain configuration
    (configuration.pyx's logic - the grain -> grain collision mapping),
    the CSE collision model (cse.pyx - the eigenvalue lumping: the
    collision operator in the CSE basis; the allen + georgievskii
    variants per the method strings).
    Plugs into step 4's network via the collision protocol (the network's
    rate-matrix off-diagonal calls the collision model).
- tests/test_collision_cse.py:
    the collision frequency: LJ params + bath -> the frequency vs RMG-Py
    (a hand-set case - the formula check);
    the CSE lumping: on step 4's toy Lindemann network, the full k(T,P)
    (with CSE) vs RMG-Py (the gate's propane case is the real test; this
    is the pre-gate proof that CSE + the network compose correctly);
    the configuration: the grain mapping matches RMG-Py's.
- The MSC/RS/SLS are NOT ported here (job 08) - the network's dispatch
  (step 4) raises NotImplemented for them; do not port them early.

## Checks (must run and pass before you claim done)

  pytest tests/test_collision_cse.py -q -> all pass
    the toy Lindemann k(T,P) with CSE vs RMG-Py (max rel diff recorded -
    this is the full-pipeline proof before the gate)

## Pitfalls

- CSE is the default + the first port (per the plan) - MSC/RS/SLS
  share this network/DoS core (only the collision lumping differs) - do
  not duplicate the network code for them (job 08 adds the methods onto
  this core).
- The collision frequency's LJ params come from the transport DB (job
  02) - if a species has no LJ entry, RMG uses a correlation/default -
  port that fallback (document it) - a missing param crashing the ME is a
  job-08/10 blocker.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-07/step-05: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-07/step-05 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-07-step-05-collision.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
