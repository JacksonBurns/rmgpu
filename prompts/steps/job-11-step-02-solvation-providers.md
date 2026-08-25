# job-11/step-02: Solvation thermo + kinetics providers

Job: job-11 - Plugin protocol + solvation plugin + liquid reactors
Prereq: job-10 done (gas-phase parity green)
This step is part of that job. The job's overall goal:
1. The RmgpuPlugin protocol (PLAN.md 9.1), wired into the core loop (the core stays agnostic to which plugins are loaded). 2. The SOLVATION plugin: the solvated thermo/kinetics providers + the LiquidReactor/MBSampledReactor, so `rmgpu run` on a liquid-phase example works. 3. A working liquid-phase parity example. The solvation plugin is the LIGHT plugin - it validates the protocol before the heavy one (catalysis, job 12).
The job's gate (run by the job's final step):
gates/gate_11.py: 1. Protocol test: a trivial no-op plugin (register_* no-ops) loads + a gas-phase run (superminimal) is byte-identical to the no-plugin run (proves the hooks are inert when unimplemented - the core is agnostic). 2. Solvation thermo parity: 25 solvated species: rmgpu solvated thermo == RMG-Py solvated thermo (the gas value + the SMD correction), tolerance 1e-8. The correction terms themselves (the delta G solv) match, not just the sum (catches sign/unit bugs). 3. Liquid reactor parity: the `liquid_phase` example in rmgpu vs RMG-Py: the final mechanism sets (the core species/reaction D, same as job 10) + the observable (conversion vs time) max diff. `liquid_phase_constSPC` too if time permits. The liquid_oxidation regression input: the mechanism-set parity. 4. MB-sampled: an MB-sampled example (the RMS_ ones in test/regression): the reactor profiles match RMG-Py within the solver tolerance. 5. Plugin isolation: the solvation plugin's code touches ONLY rmgpu/plugins/ + the core hook call-sites (verify by diff: no solvation code in rmgpu/core, rmgpu/ml, rmgpu/reactor).

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

- Previous step (already done, its code is in the tree): job-11/step-01-protocol
- This step: job-11/step-02-solvation-providers
- Next step (do NOT start it): job-11/step-03-liquid-reactors

## Goal

The solvation plugin's providers: the solvated-thermo provider (the
gas value + the solvation free-energy correction) + the solvated-kinetics
provider (the rate correction) - ported from RMG's data/solvation.py
(SMD-like corrections, the solvation DB).

## Reference to read (this step's budget)

  RMG-Py/rmgpy/data/solvation.py  (2455 - the implicit-solvation
    (SMD-like) corrections parameterized in the solvation DB; the
    solvated-thermo + the rate-correction logic - port the semantics)
  /home/jackson/rmgpu/RMG-database/input/solvation/ (the solvation
    groups/libraries - the data; rmgdb's solvation.db has the tables -
    job-02's SolvationDB facade)
  (job-04's estimators - the gas-phase thermo/kinetics the correction is
    added to; job-02's SolvationDB)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/plugins/solvation/thermo.py:
    `SolvationThermoProvider` (implements register_thermo): for a species
    in solution (the YAML solvation: block present), returns the solvated
    thermo = the gas-phase thermo (ML/library - job 04) + the solvation
    free-energy correction (from the solvation DB via the SolvationDB
    facade - the SMD-like correction per data/solvation.py's semantics).
    The provider is a (gas_thermo, solvation_db) -> solvated_thermo
    function the core's thermo resolver consults when the run is in
    solution.
- rmgpu/plugins/solvation/kinetics.py:
    `SolvationKineticsProvider` (implements register_kinetics): the
    solvation correction to the rates (the reaction free-energy-of-
    solvation shifts the barrier - the correction per data/solvation.py)
    - the same DB.
- The correction terms are exposed (the delta G solv per species/reaction)
  - the gate checks the TERMS match (not just the sum - catches sign/unit
  bugs).
- tests/test_solvation_providers.py: 25 solvated species: the solvated
  thermo (the gas + the correction) vs RMG-Py (the correction terms + the
  sum - 1e-8); a solvated reaction: the rate correction vs RMG-Py (the
  barrier shift).

## Checks (must run and pass before you claim done)

  pytest tests/test_solvation_providers.py -q -> all pass
  the 25 species: the correction terms + the sum vs RMG-Py (1e-8)

## Pitfalls

- The solvation correction is a PROVIDER (it wraps the gas-phase
  value - job 04's) - not a new thermo engine (the core composes: the
  gas thermo from the resolver + the correction from the provider).
- The SMD-like correction is RMG's (data/solvation.py) - port the
  semantics + the DB parameters; verify against RMG-Py's numbers (the
  gate) - do not derive your own.
- The provider is active ONLY when the run is in solution (the solvation:
  block) - a gas-phase run is unaffected (the provider is inert).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-11/step-02: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-11/step-02 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-11-step-02-solvation-providers.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
