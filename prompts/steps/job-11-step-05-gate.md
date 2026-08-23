# job-11/step-05: Job-11 gate (protocol + solvation + liquid parity)

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

- Previous step (already done, its code is in the tree): job-11/step-04-plugin
- This step: job-11/step-05-gate
- Next step (do NOT start it): the job gate for job-11 (its step file)

## Goal

Run the job-11 gate: the protocol test (the no-op plugin is inert),
the solvation thermo parity (25 species, the terms + the sum), the liquid
reactor parity (liquid_phase + liquid_phase_constSPC + liquid_oxidation),
the MB-sampled example, the plugin isolation. Close job 11.

## Reference to read (this step's budget)

  (no new reference reads)
  RMG-Py/examples/rmg/{liquid_phase,liquid_phase_constSPC}/input.py +
    test/regression/{liquid_oxidation,RMS_CSTR_liquid_oxidation}/ (the
    liquid examples - the MB-sampled one located from the RMS_ set)
  (the harness from job 10/1 - the parity_example.py protocol, reused for
    the liquid examples)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- gates/gate_11.py: the five checks (the protocol test: the NoOp
  plugin + the superminimal run byte-identical to the no-plugin run; the
  solvation thermo parity: 25 species, the correction terms + the sum,
  1e-8; the liquid reactor parity: liquid_phase (the mechanism sets + the
  observable) + liquid_phase_constSPC + liquid_oxidation (the mechanism
  sets); the MB-sampled: an RMS_ example (the profiles within the solver
  tolerance); the plugin isolation: the diff - no solvation code in
  rmgpu/core, rmgpu/ml, rmgpu/reactor).
- The RMG-Py references: the 25 solvated species (the correction terms +
  the sum), the liquid examples (the mechanism sets + the profiles) ->
  gates/baselines/job11/ (commit).
- reports/job-11.md: the plugin protocol as-implemented (any deviation
  from PLAN.md 9.1 + the PLAN.md update if there is one), the five checks'
  numbers (real), the liquid-phase basis verification (the mole fraction
  vs the activity - confirmed against RMG-Py).
- A RED gate: the liquid-phase basis is the prime suspect (the mole-
  fraction-vs-activity bug) - root-cause to the reactor (step 11/3) or
  the providers (step 11/2).

## Checks (must run and pass before you claim done)

  /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_11.py
    -> the five checks recorded (targets: the byte-identical no-op run,
    1e-8 on the solvation terms, the liquid mechanism-set parity, the
    MB-sampled within the tolerance, the isolation clean)
  pytest tests/ -q -> all pass

## Pitfalls

- The protocol test (check 1) is the foundation - a difference
  between the no-op-plugin run and the no-plugin run is a core bug (the
  hook call-sites are not inert), not a plugin bug; fix the core (step
  11/1) before anything else.
- The plugin isolation (check 5) is a DESIGN invariant (the plugin is a
  bag of providers + hooks, not a fork - PLAN.md 9) - a solvation symbol
  in rmgpu/core is a violation; refactor it into the plugin.
- The liquid parity is the job's payoff (the first non-gas-phase parity) -
  the mechanism-set parity target is the same as job 10 (the core
  identical or documented).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-11/step-05: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-11/step-05 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-11-step-05-gate.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
