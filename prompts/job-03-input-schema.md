# job-03: YAML input schema + CLI + legacy .py importer

Status of this file: a JOB BRIEF, not a task. Do not try to "do this job"
in one session - it is decomposed into the steps below, one session each.
The coordinator (see README.md, "Session vs step") picks the next step from
STATUS.md's NEXT pointer; this file is the map of the job.

## Goal

Replace RMG's "execute a Python file to configure the run" with a declarative, schema-validated YAML document + a small CLI + a lossless importer for the legacy .py DSL. This is the user-facing front door. Deliverables: rmgpu/schemas/input.py (pydantic), rmgpu/cli.py (click), rmgpu/importer/legacy.py (ast-based).

## Prereq

job-01 done (Molecule for structure parsing). job-02 not strictly required, but the database: block references library names from it.

## Steps (strictly sequential; one fresh subagent session each)

  step 01  prompts/steps/job-03-step-01-core.md  Input schema: core blocks (quantity, database, species, forbidden)
  step 02  prompts/steps/job-03-step-02-blocks.md  Input schema: reactors + remaining blocks + extends
  step 03  prompts/steps/job-03-step-03-cli.md  CLI: run/validate/schema/version
  step 04  prompts/steps/job-03-step-04-legacy.md  Legacy importer: inventory + ast visitor
  step 05  prompts/steps/job-03-step-05-gate.md  Job-03 gate (lossless import of 47 examples)

## The job gate

Run by the final step's session (gates/gate_03.py, report
reports/job-03.md):

gates/gate_03.py: 1. DSL inventory: 47 legacy input.py files x functions used (report table). 2. Import: N of 47 files import to schema-valid YAML with zero dropped values; the rest have documented IMPORT-NOTEs. Target: N == 47. 3. `rmgpu validate` on all 47 imported YAMLs: all pass. 4. `rmgpu run minimal.yaml` (the imported minimal example) prints the resolved document. 5. JSON schema exports and validates a hand-written minimal input.yaml (the example from PLAN.md 12.2).

## When the job is done

The final step's report (reports/job-03.md) has the gate result, the
job table row is `done` (or `blocked` with the cause), and the NEXT pointer
in STATUS.md targets job-04's first step (if there is a next job).
