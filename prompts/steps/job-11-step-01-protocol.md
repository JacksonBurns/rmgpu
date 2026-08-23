# job-11/step-01: The plugin protocol (base + the core hook call-sites)

Job: job-11 - Plugin protocol + solvation plugin + liquid reactors
Prereq: job-10 done (gas-phase parity green)
This step is part of that job. The job's overall goal:
1. The RmgpuPlugin protocol (PLAN.md 9.1), wired into the core loop (the core stays agnostic to which plugins are loaded). 2. The SOLVATION plugin: the solvated thermo/kinetics providers + the LiquidReactor/MBSampledReactor, so `rmgpu run` on a liquid-phase example works. 3. A working liquid-phase parity example. The solvation plugin is the LIGHT plugin - it validates the protocol before the heavy one (catalysis, job 12).
The job's gate (run by the job's final step):
gates/gate_11.py: 1. Protocol test: a trivial no-op plugin (register_* no-ops) loads + a gas-phase run (superminimal) is byte-identical to the no-plugin run (proves the hooks are inert when unimplemented - the core is agnostic). 2. Solvation thermo parity: 25 solvated species: rmgpu solvated thermo == RMG-Py solvated thermo (the gas value + the SMD correction), tolerance 1e-8. The correction terms themselves (the delta G solv) match, not just the sum (catches sign/unit bugs). 3. Liquid reactor parity: the `liquid_phase` example in rmgpu vs RMG-Py: the final mechanism sets (the core species/reaction D, same as job 10) + the observable (conversion vs time) max diff. `liquid_phase_constSPC` too if time permits. The liquid_oxidation regression input: the mechanism-set parity. 4. MB-sampled: an MB-sampled example (the RMS_ ones in test/regression): the reactor profiles match RMG-Py within the solver tolerance. 5. Plugin isolation: the solvation plugin's code touches ONLY rmgpu/plugins/ + the core hook call-sites (verify by diff: no solvation code in rmgpu/core, rmgpu/ml, rmgpu/reactor).

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
- This step: job-11/step-01-protocol
- Next step (do NOT start it): job-11/step-02-solvation-providers

## Goal

The RmgpuPlugin ABC (PLAN.md 9.1) + the plugin loader (the
entry-point discovery) + the core hook call-sites (the core loop gets the
hooks - small + unconditional - the core stays agnostic to which plugins
are loaded).
The protocol is the deliverable - get the hook semantics EXACTLY as
PLAN.md 9.1 (they were designed so solvation = 2 providers + reactors,
catalysis = all of them).

## Reference to read (this step's budget)

  /home/jackson/rmgpu/rmgpu/PLAN.md 9.1 (the EXACT protocol:
    register_species/families/thermo/kinetics/reactor +
    on_new_species/on_new_reaction/after_prune + load_database)
  (job-06/2's model.py - the core loop - the hook call-sites go here)
  (job-03's schema - the plugins: block, if present; if not, the minimal
    way to enable a plugin)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/plugins/base.py:
    `RmgpuPlugin` ABC per PLAN.md 9.1 (the registration methods:
    register_species, register_families, register_thermo, register_
    kinetics, register_reactor; the per-object hooks: on_new_species,
    on_new_reaction, after_prune; the data: load_database).
    the plugin loader: the entry-point discovery (plugins register via
    the `rmgpu.plugins` entry-point group; the loader discovers +
    instantiates them + calls the register_* at model init).
    a NoOpPlugin (the trivial no-op plugin - the gate's protocol test
    uses it).
- The core hook call-sites (job-06/2's model.py): the core loop calls the
  hooks in the fixed order (the register_* at init; on_new_species after a
  species is added; on_new_reaction after a reaction is estimated;
  after_prune after a prune) - the call-sites are SMALL + UNCONDITIONAL
  (they loop over the loaded plugins; an empty plugin list is a no-op).
  The core NEVER knows which plugins are loaded (it calls the hooks; the
  plugins decide what to do).
- A minimal way to enable a plugin in the YAML (if job 03's schema has no
  plugins: block, add the minimum: a plugins: list of plugin names -
  note it in the report; the schema is the API, a new top-level key is a
  version-bump candidate - record it).
- tests/test_plugin_protocol.py: the NoOpPlugin loads + a superminimal run
  with it is IDENTICAL to the no-plugin run (the hook call-sites are inert
  when unimplemented - the core is agnostic); the loader discovers a
  registered plugin (a test plugin via the entry-point).

## Checks (must run and pass before you claim done)

  pytest tests/test_plugin_protocol.py -q -> all pass
  superminimal: the no-plugin run == the NoOpPlugin run (byte-identical
    core/edge + the output tree)

## Pitfalls

- The protocol is the deliverable (PLAN.md 9.1) - a deviation from
  the protocol is a PLAN.md 9 change (the catalysis job builds on it) -
  if reality forces a change, update PLAN.md 9.1 (note it in the report).
- The hook call-sites are SMALL + UNCONDITIONAL - a core that special-
  cases a plugin (an `if plugin == solvation`) is a protocol violation
  (the core must stay agnostic).
- The NoOpPlugin run MUST be byte-identical to the no-plugin run (the
  hooks must be truly inert) - a difference is a core bug (the call-sites
  changed something even with no plugin logic).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-11/step-01: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-11/step-01 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-11-step-01-protocol.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
