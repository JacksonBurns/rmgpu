# job-11: Plugin protocol + solvation plugin + liquid reactors

Status of this file: a JOB BRIEF, not a task. Do not try to "do this job"
in one session - it is decomposed into the steps below, one session each.
The coordinator (see README.md, "Session vs step") picks the next step from
STATUS.md's NEXT pointer; this file is the map of the job.

## Goal

1. The RmgpuPlugin protocol (PLAN.md 9.1), wired into the core loop (the core stays agnostic to which plugins are loaded). 2. The SOLVATION plugin: the solvated thermo/kinetics providers + the LiquidReactor/MBSampledReactor, so `rmgpu run` on a liquid-phase example works. 3. A working liquid-phase parity example. The solvation plugin is the LIGHT plugin - it validates the protocol before the heavy one (catalysis, job 12).

## Prereq

job-10 done (gas-phase parity green)

## Steps (strictly sequential; one fresh subagent session each)

  step 01  prompts/steps/job-11-step-01-protocol.md  The plugin protocol (base + the core hook call-sites)
  step 02  prompts/steps/job-11-step-02-solvation-providers.md  Solvation thermo + kinetics providers
  step 03  prompts/steps/job-11-step-03-liquid-reactors.md  The LiquidReactor + MBSampledReactor
  step 04  prompts/steps/job-11-step-04-plugin.md  The solvation plugin assembly + the liquid parity example
  step 05  prompts/steps/job-11-step-05-gate.md  Job-11 gate (protocol + solvation + liquid parity)

## The job gate

Run by the final step's session (gates/gate_11.py, report
reports/job-11.md):

gates/gate_11.py: 1. Protocol test: a trivial no-op plugin (register_* no-ops) loads + a gas-phase run (superminimal) is byte-identical to the no-plugin run (proves the hooks are inert when unimplemented - the core is agnostic). 2. Solvation thermo parity: 25 solvated species: rmgpu solvated thermo == RMG-Py solvated thermo (the gas value + the SMD correction), tolerance 1e-8. The correction terms themselves (the delta G solv) match, not just the sum (catches sign/unit bugs). 3. Liquid reactor parity: the `liquid_phase` example in rmgpu vs RMG-Py: the final mechanism sets (the core species/reaction D, same as job 10) + the observable (conversion vs time) max diff. `liquid_phase_constSPC` too if time permits. The liquid_oxidation regression input: the mechanism-set parity. 4. MB-sampled: an MB-sampled example (the RMS_ ones in test/regression): the reactor profiles match RMG-Py within the solver tolerance. 5. Plugin isolation: the solvation plugin's code touches ONLY rmgpu/plugins/ + the core hook call-sites (verify by diff: no solvation code in rmgpu/core, rmgpu/ml, rmgpu/reactor).

## When the job is done

The final step's report (reports/job-11.md) has the gate result, the
job table row is `done` (or `blocked` with the cause), and the NEXT pointer
in STATUS.md targets job-12's first step (if there is a next job).
