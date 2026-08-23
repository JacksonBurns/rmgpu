# job-11/step-03: The LiquidReactor + MBSampledReactor

Job: job-11 - Plugin protocol + solvation plugin + liquid reactors
Prereq: job-10 done (gas-phase parity green)
This step is part of that job. The job's overall goal:
1. The RmgpuPlugin protocol (PLAN.md 9.1), wired into the core loop (the core stays agnostic to which plugins are loaded). 2. The SOLVATION plugin: the solvated thermo/kinetics providers + the LiquidReactor/MBSampledReactor, so `rmgpu run` on a liquid-phase example works. 3. A working liquid-phase parity example. The solvation plugin is the LIGHT plugin - it validates the protocol before the heavy one (catalysis, job 12).
The job's gate (run by the job's final step):
gates/gate_11.py: 1. Protocol test: a trivial no-op plugin (register_* no-ops) loads + a gas-phase run (superminimal) is byte-identical to the no-plugin run (proves the hooks are inert when unimplemented - the core is agnostic). 2. Solvation thermo parity: 25 solvated species: rmgpu solvated thermo == RMG-Py solvated thermo (the gas value + the SMD correction), tolerance 1e-8. The correction terms themselves (the delta G solv) match, not just the sum (catches sign/unit bugs). 3. Liquid reactor parity: the `liquid_phase` example in rmgpu vs RMG-Py: the final mechanism sets (the core species/reaction D, same as job 10) + the observable (conversion vs time) max diff. `liquid_phase_constSPC` too if time permits. The liquid_oxidation regression input: the mechanism-set parity. 4. MB-sampled: an MB-sampled example (the RMS_ ones in test/regression): the reactor profiles match RMG-Py within the solver tolerance. 5. Plugin isolation: the solvation plugin's code touches ONLY rmgpu/plugins/ + the core hook call-sites (verify by diff: no solvation code in rmgpu/core, rmgpu/ml, rmgpu/reactor).

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

- Previous step (already done, its code is in the tree): job-11/step-02-solvation-providers
- This step: job-11/step-03-liquid-reactors
- Next step (do NOT start it): job-11/step-04-plugin

## Goal

The liquid-phase reactors: the LiquidReactor (the Nernst/liquid-
phase mole balances + the liquid mass-transfer coefficient power law) +
the MBSampledReactor (the Monte-Carlo rate sampling) - ported onto the
torchdae backend. Registered via the plugin's register_reactor.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/solver/liquid.pyx  (758 - the LiquidReactor: the
    Nernst, the liquid-phase mole balances, the liquid mass-transfer
    coefficient power law from the input DSL - port the MATH)
  RMG-Py/rmgpy/solver/mbSampled.pyx  (505 - the MB-sampled reactor: the
    Monte-Carlo sampling of the rate (RMG's own method, NOT a standard
    solver) - port the exact sampling)
  (job-06/1's torch backend - the reactors plug into it; the termination
    criteria from job 06/1)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/plugins/solvation/reactors.py:
    `LiquidReactor` (port liquid.pyx's math onto the torchdae backend -
    the liquid-phase mole balances, the Nernst, the mass-transfer
    coefficient power law from the input DSL): the liquid basis (the mole
    FRACTION, not the partial pressure - activities; the classic bug is
    the mole-fraction-vs-activity confusion - port RMG's liquid basis
    EXACTLY).
    `MBSampledReactor` (port mbSampled.pyx - the Monte-Carlo sampling of
    the rate - RMG's exact method, NOT a standard solver - port the
    sampling; it is stochastic, so the parity is within the solver
    tolerance + a fixed seed).
    Both registered via the plugin's register_reactor (the core's reactor
    factory gets them).
- The YAML wiring (job 03's schema): the liquid + mb_sampled reactor
  types (job 03 defined them - the plugin makes them LIVE); the
  solvation: block enables the liquid-phase run (the reactors are
  constrained to the liquid types when present).
- tests/test_liquid_reactors.py: a liquid-phase toy system (a 2-species
  liquid reaction): the profiles vs RMG-Py (the liquid basis - the mole
  fractions, the conversion - within the solver tolerance); the MB-
  sampled: the sampling is stochastic - a fixed seed + the profiles
  within the tolerance (the RMG-Py reference with the same seed, if
  RMG's is seedable - else the statistical parity: the mean over a few
  runs).

## Checks (must run and pass before you claim done)

  pytest tests/test_liquid_reactors.py -q -> all pass
  the liquid toy: the profiles vs RMG-Py (within the solver tolerance)

## Pitfalls

- The liquid basis (the mole fraction, the activities) is the
  classic bug - port RMG's EXACTLY (a mole-fraction-vs-activity
  confusion shifts every liquid profile); verify against RMG-Py (the
  gate).
- The MB-sampled reactor is STOCHASTIC (the Monte-Carlo sampling) - the
  parity is within the tolerance (a fixed seed where possible; else the
  statistical parity) - do NOT expect a deterministic match.
- The reactors plug into job 06/1's torch backend (the ODE rhs is the
  liquid-phase balance) - do NOT fork the solver; the backend is
  shared.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-11/step-03: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-11/step-03 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-11-step-03-liquid-reactors.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
