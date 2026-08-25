# job-11/step-04: The solvation plugin assembly + the liquid parity example

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

- Previous step (already done, its code is in the tree): job-11/step-03-liquid-reactors
- This step: job-11/step-04-plugin
- Next step (do NOT start it): job-11/step-05-gate

## Goal

Assemble the solvation plugin (the providers + the reactors + the
load_database + the hooks) + wire the YAML solvation: block + run the
liquid-phase parity example (liquid_phase) end to end.

## Reference to read (this step's budget)

  (the providers from step 11/2; the reactors from step 11/3; the
    protocol from step 11/1)
  RMG-Py/examples/rmg/liquid_phase/input.py (the liquid-phase example -
    the parity target)
  (job-02's SolvationDB - the load_database pulls the solvation
    groups/libraries)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/plugins/solvation/plugin.py:
    `SolvationPlugin(RmgpuPlugin)`: the assembly - register_thermo
    (step 11/2's provider), register_kinetics (step 11/2's provider),
    register_reactor (step 11/3's LiquidReactor + MBSampledReactor),
    load_database (the solvation groups/libraries from the SolvationDB),
    on_new_species (the liquid-phase species constraints - which species
    may be in the liquid - RMG's liquid-phase screening),
    on_new_reaction/after_prune (the no-ops where solvation adds nothing).
    the entry-point registration (the plugin is discoverable via the
    `rmgpu.plugins` group - the loader finds it).
- The YAML wiring: the solvation: block (job 03's schema) enables the
  plugin (when present, the run is liquid-phase: the reactors constrained
  to the liquid types, the solvation providers active).
- The liquid_phase example: import it (job 03), run it in rmgpu (the
  plugin active), the output tree (the liquid-phase profiles).
- reports/job-11-step-04.md: the plugin as-  implemented (any deviation from PLAN.md 9.1 - there should be none; if
  there is, explain + update PLAN.md), the liquid_phase run (the iteration
  count, the core/edge counts, the profiles).
- The full tests/ suite green.

## Checks (must run and pass before you claim done)

  pytest tests/ -q -> all pass
  `rmgpu run` on the imported liquid_phase input: completes (the plugin
    active, the liquid reactors, the solvation providers)

## Pitfalls

- The plugin is the DELIVERABLE validation of the protocol (PLAN.md
  9.3: solvation is the reference for how LIGHT a plugin can be) - if the
  assembly forces a protocol change, that is a protocol finding (update
  PLAN.md 9.1), not a silent workaround.
- The liquid_phase example is the parity target (the gate, step 11/5) -
  this step makes it RUN; the gate measures it against RMG-Py.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-11/step-04: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-11/step-04 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-11-step-04-plugin.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
