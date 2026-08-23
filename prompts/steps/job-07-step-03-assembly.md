# job-07/step-03: Conformer assembly from the statmech DB (no QM)

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

- Previous step (already done, its code is in the tree): job-07/step-02-torsion
- This step: job-07/step-03-assembly
- Next step (do NOT start it): job-07/step-04-network

## Goal

The conformer assembly: how rmgpu builds a Conformer for a species
WITHOUT QM - the statmech DB path (group characteristic frequencies + the
heat-capacity fitting). This is THE data path that makes pdep work without
QM (PLAN.md 8a.2).

## Reference to read (this step's budget)

  RMG-Py/rmgpy/data/statmech.py  (743 - get_statmech_data (lines
    ~318-460): the group frequencies + the remainder fitted to the
    species' heat capacity - the algorithm to port; GroupFrequencies; the
    data flow from the statmech DB)
  RMG-Py/rmgpy/data/statmechfit.py  (537 - the heat-capacity fitting
    (statmechfit): the algorithm that fits the group frequencies to the
    species' Cp (from the thermo model) - port it)
  /home/jackson/rmgpu/rmgdb/standard/rmgdb/statmech/schema.py (the
    statmech tables - the group frequency data)
  (job-02's StatMechDB facade; job-04's thermo for the Cp target)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/data/statmech.py:
    `get_statmech_data(molecule, statmech_db, thermo) -> Conformer` (the
    rmgpu home of RMG-Py's data/statmech.py algorithm):
      the group characteristic frequencies (from the statmech DB - the
      GroupFrequencies: e.g. "ringCH -> 2750-3150 cm^-1 x 2") assigned to
      the molecule's atoms/bonds per RMG's assignment rules;
      the remainder (the modes the groups don't cover) FITTED to the
      species' heat capacity (statmechfit's algorithm - the Cp target is
      the ML/library thermo from job 04 - the fitting makes the total Cp
      match the thermo model's Cp(T), per RMG);
      the Conformer assembled (E0 from the thermo model's Hf298 - the ML
      value, step 6's TS-E0 path uses it; spin from the Molecule; the
      modes from steps 1-2; the symmetry number from job 01).
    This is database + fitting logic, NOT estimation - it is the
    frequency path (PLAN.md 8a.2: "frequencies <- statmech DB group freqs
    [no QM]").
- tests/test_statmech_assembly.py: for 20 species (incl. intermediates -
  the ones pdep actually needs): the assembled Conformer (E0, spin, mode
  counts, the group-assignment) matches RMG-Py's get_statmech_data
  (reference script - the gate's 50-species set generalizes it); the total
  Cp(T) (assembled conformer) matches the thermo model's Cp(T) (the fit's
  purpose - within the fit tolerance RMG uses).

## Checks (must run and pass before you claim done)

  pytest tests/test_statmech_assembly.py -q -> all pass
  the 20-species assembly: E0/spin/mode-counts match RMG-Py; the Cp fit
    closes (assembled Cp vs thermo-model Cp within the fit tolerance)

## Pitfalls

- E0 comes from the thermo model (Hf298) - in rmgpu that is the ML
  value (or the library value if the species is in a library - the
  resolver, job 04). Do NOT introduce a QM path for E0 - the data path is
  PLAN.md 8a.2 and it has no QM.
- The group-assignment rules (which group frequency goes to which atom/
  bond) are RMG's exact rules - port them; a wrong assignment shifts the
  DoS (the gate's DoS sub-gate catches it).
- The Cp fit (statmechfit) is iterative (it adjusts the remainder modes) -
  port RMG's convergence criterion; a non-converging fit is a recorded
  finding, not a silent skip.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-07/step-03: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-07/step-03 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-07-step-03-assembly.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
