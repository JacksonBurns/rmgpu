# job-07/step-04: The pdep network + master equation (CSE) + TS-E0

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

- Previous step (already done, its code is in the tree): job-07/step-03-assembly
- This step: job-07/step-04-network
- Next step (do NOT start it): job-07/step-05-collision

## Goal

The network + the master equation itself: the Network (isomers,
channels, grains, the rate matrix), the time-dependent ME integration
(torchdae), the k(T,P) extraction, and the TS-E0 derivation from the HPL
rate (the no-QM path). CSE collision (step 5) plugs into this.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/pdep/network.py  (1149 - the Network: grains, the
    rate matrix (RRKM k(E) from the TS DoS + the energy gap), the time-
    dependent ME integration, the k(T,P) phenomenological-rate extraction
    (the quasi-stationary flux), the grain generation (max grain size, min
    count, the energy_grid logic), the method dispatch (the LONG method
    strings - the table in the job brief))
  RMG-Py/rmgpy/pdep/me.pyx  (186 - the time-dependent ME ODE - port to
    torchdae: it is a stiff ODE in the grain populations; use the job-06
    backend)
  RMG-Py/rmgpy/pdep/reaction.pyx  (402 - the pressure-dependent reaction
    object: isomers/channels - the data structure the network consumes)
  RMG-Py/arkane/pdep.py  (lines ~270-350 - the TS E0 DERIVATION from the
    HPL rate (the no-QM path): E0_TS = sum(reactant E0) - R*T*ln(k_inf*V/h)
    - port EXACTLY, including the tunneling parameter handling (keep
    Wigner from job 04/4))
  (the statmech conformer from step 3; the rate registry from job 02/04)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/pdep/network.py:
    `Network`: isomers (Conformers from step 3), reactant/product channels
    (the pdep reaction object from reaction.pyx's semantics), the bath gas,
    the grains (generate per RMG's grain rules: max grain size, min count,
    the energy_grid logic), the rate matrix (RRKM k(E) diagonal from the
    TS DoS (step 1's get_number_of_states) + the energy gap to the TS;
    the collision off-diagonal - a protocol the collision model implements
    (step 5's CSE; a stub that raises until then)), the time-dependent ME
    integration via torchdae (the me.pyx port - stiff; BDF2/TR-BDF2 per
    job-06's choice), the k(T,P) extraction (the phenomenological rate
    from the quasi-stationary flux - port network.py's extraction).
    The method dispatch on the LONG strings (port exactly):
      'modified strong collision' -> MSC (job 08 - raise NotImplemented
      until then)
      'reservoir state' -> RS (job 08)
      'chemically-significant eigenvalues' -> CSE (allen)
      'chemically-significant eigenvalues georgievskii' -> CSE (georgievskii)
      'simulation least squares' / '...ode' / '...matrix exponential'
      -> SLS (job 08)
    CSE (allen) is THIS job (step 5's collision + this network).
- `derive_ts_e0(reactant_e0s, k_inf, T, V, tunneling) -> E0_TS` (the
  arkane/pdep.py no-QM path, ported exactly): E0_TS = sum(reactant E0) -
  R*T*ln(k_inf*V/h) - the tunneling parameter handling (Wigner from job
  04) per RMG.
- tests/test_pdep_network.py:
    the TS-E0 derivation: a known Lindemann case must reproduce RMG's
    k(T,P) (a standalone test - a single isomer + one channel, the
    parameters hand-set, k(T,P) at a grid vs RMG-Py's network on the same
    params - THIS test must pass before the driver is wired);
    the grain generation: a conformer's grain grid matches RMG-Py's
    (counts + the grain boundaries);
    the ME integration: the toy Lindemann k(T,P) curve vs RMG-Py (the
    sub-gate that de-risks the k(T,P) gate before the full example).
- The network consumes the conformer (step 3), the TS E0 (this step), and
  the collision model (step 5) - the wiring is the protocol, so step 5
  plugs in without touching the network.

## Checks (must run and pass before you claim done)

  pytest tests/test_pdep_network.py -q -> all pass
  the TS-E0 derivation: the known Lindemann case reproduces RMG's k(T,P)
    (the max rel diff recorded - this is the no-QM path's proof)
  the grain grid matches RMG-Py (counts + boundaries)

## Pitfalls

- The RRKM k(E) (from the TS DoS + the energy gap) is the rate
  matrix's diagonal - a wrong DoS (step 1) or a wrong TS E0 (this step)
  shifts EVERY k(T,P). The toy-Lindemann sub-gate (this step) isolates
  them from the driver's complexity.
- The ME is a stiff ODE in the grain populations (collision frequencies
  vary by orders of magnitude across grains) - torchdae BDF2/TR-BDF2;
  validate against RMG-Py's integration (the toy case covers it).
- The TS-E0 derivation is THE no-QM path (PLAN.md 8a.2) - it must match
  RMG-Py's exactly (the gate's sub-gate 2 is 1e-8). A deviation here is a
  BLOCKER (it propagates to every pressure-dependent k(T,P)).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-07/step-04: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-07/step-04 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-07-step-04-network.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
