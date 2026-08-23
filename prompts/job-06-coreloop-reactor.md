# job-06: Core/edge mechanism loop + torchdae reactor (first integration)

Status of this file: a JOB BRIEF, not a task. Do not try to "do this job"
in one session - it is decomposed into the steps below, one session each.
The coordinator (see README.md, "Session vs step") picks the next step from
STATUS.md's NEXT pointer; this file is the map of the job.

## Goal

The `rmgpu run` command actually works end to end (gas phase, constant T or T/P): load YAML -> build model from seed species -> estimate properties (libraries/ML) -> enumerate candidate reactions (recipes) -> simulate in torchdae -> screen by conversion -> grow/prune -> iterate to steady state -> write the output tree (PLAN.md 12.3), including the canonical `mechanism/core.yaml`. The moment of truth for the stack.

## Prereq

jobs 01-05 done (all of: molecule, db, schema, estimators, recipes)

## Steps (strictly sequential; one fresh subagent session each)

  step 01  prompts/steps/job-06-step-01-reactor.md  Reactor definitions + termination + torchdae backend
  step 02  prompts/steps/job-06-step-02-model.md  CoreEdgeReactionModel (enlarge/prune/screen)
  step 03  prompts/steps/job-06-step-03-driver.md  main.py: the job driver + the iteration loop
  step 04  prompts/steps/job-06-step-04-output.md  Mechanism artifact schema + the output tree writer
  step 05  prompts/steps/job-06-step-05-chemkin.md  Chemkin writer + species dictionary
  step 06  prompts/steps/job-06-step-06-gate.md  Job-06 gate (first real mechanism generation)

## The job gate

Run by the final step's session (gates/gate_06.py, report
reports/job-06.md):

gates/gate_06.py: 1. torchdae sub-gate: a stiff reference ODE (a 3-reaction Lindemann falloff toy system, or Van der Pol mu=10 as a non-chemistry control) integrated by torchdae vs a high-accuracy reference (RK45 tiny step) - max abs diff recorded (PLAN.md 13 risk 3: prove it before trusting it with mechanism growth). 2. `rmgpu run` on the imported `superminimal` (HPL kinetics, pdep off/stub): completes to steady state (or max_iter); iteration count + core/edge species+reaction counts recorded. 3. Parity vs RMG-Py (pdep OFF, same input; its output -> gates/baselines/superminimal/): |core_rmgpu - core_rmg| and |core+edge_rmgpu - (core+edge)_rmg| as sets. TARGET: core identical (or documented divergence with cause); edge within a small fraction. Any systematic divergence (a family always missing/extra): fix if tractable in this job, else record precisely. 4. Same for `c3h4` (the right size - NOT the GRI-scale example). 5. Output tree: every file in PLAN.md 12.3 exists and is valid (YAML parses, CSV columns right, core.yaml loads back via the schema and re-simulates the final iteration's profiles within tolerance). 6. provenance.yaml contains real hashes/versions (not placeholders).

## When the job is done

The final step's report (reports/job-06.md) has the gate result, the
job table row is `done` (or `blocked` with the cause), and the NEXT pointer
in STATUS.md targets job-07's first step (if there is a next job).
