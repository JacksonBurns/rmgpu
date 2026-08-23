# job-00: Environment, package skeleton, test scaffolding

Status of this file: a JOB BRIEF, not a task. Do not try to "do this job"
in one session - it is decomposed into the steps below, one session each.
The coordinator (see README.md, "Session vs step") picks the next step from
STATUS.md's NEXT pointer; this file is the map of the job.

## Goal

Set up the foundation every later job builds on: the `rmgpu` conda env, the package skeleton, test tooling, and the gate harness. No chemistry yet.

## Prereq

none (first job)

## Steps (strictly sequential; one fresh subagent session each)

  step 01  prompts/steps/job-00-step-01-env.md  Conda env rmgpu (all deps, rmgdb, checkpoint locations)
  step 02  prompts/steps/job-00-step-02-skeleton.md  Package skeleton + CLI stubs + test scaffolding
  step 03  prompts/steps/job-00-step-03-gate.md  Smoke test + job-00 gate

## The job gate

Run by the final step's session (gates/gate_00.py, report
reports/job-00.md):

gates/gate_00.py + tests/test_smoke.py all pass: - every rmgpu subpackage imports; `rmgpu version` prints 0.1.0 - `import rdkit, torch, chemprop, cantera, chemicals, fluids, thermo, pint, sqlalchemy, torchdae, pydantic` all work; torch is the CUDA build (torch.cuda.is_available() is True) - the CheMeleon checkpoint paths used by RMG-Py's ml_estimator DSL are LOCATED and recorded (exact paths, formats, output dims) in the report - not made to work yet (job 04), but the env must be able to load one - rmgdb is installed and importable; install method recorded

## When the job is done

The final step's report (reports/job-00.md) has the gate result, the
job table row is `done` (or `blocked` with the cause), and the NEXT pointer
in STATUS.md targets job-01's first step (if there is a next job).
