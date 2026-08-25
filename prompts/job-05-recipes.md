# job-05: Reaction recipe DSL + product enumeration

Status of this file: a JOB BRIEF, not a task. Do not try to "do this job"
in one session - it is decomposed into the steps below, one session each.
The human reads STATUS.md's NEXT pointer; this file is the map of the job.

## Goal

Port RMG's reaction GENERATION machinery: the recipe DSL (atom- labeled bond operations) that, given a reaction template + a reactant structure, enumerates all valid product structures with degeneracy. Custom RMG IP (NOT SMARTS reactions - PLAN.md 4), ported faithfully - the engine that turns "a C=C exists" into concrete isomer products. Deliverables: rmgpu/core/recipe.py (the engine), rmgpu/core/template.py (template matching), rmgpu/core/family.py (loader + facade), plus the group matcher in rmgpu/molecule/group.py (partial port).

## Prereq

jobs 01-02 done (Molecule layer; family-definition storage decision from job-02 step 4)

## Steps (strictly sequential; one fresh human-started session each)

  step 01  prompts/steps/job-05-step-01-engine.md  ReactionRecipe engine (apply_recipe + labels)
  step 02  prompts/steps/job-05-step-02-products.md  Product enumeration (generate_reactions)
  step 03  prompts/steps/job-05-step-03-templates.md  Template matching + group matcher
  step 04  prompts/steps/job-05-step-04-families.md  Family loader + KineticsFamilies facade
  step 05  prompts/steps/job-05-step-05-gate.md  Job-05 gate (product enumeration parity)

## The job gate

Run by the final step's session (gates/gate_05.py, report
reports/job-05.md):

gates/gate_05.py over a fixed set of (family, reactants) cases (every family in the 'default' set that appears in the c3h4/superminimal mechanisms + the RMG-Py test fixtures): 1. Product enumeration parity: for each case, RMG-Py's family.get_products/apply vs rmgpu's: the SETS of products (canonical SMILES) equal, and the degeneracy per product matches EXACTLY. Report: cases, pass/fail, mismatches listed. 2. Reverse: for enumerated products, the reverse-reaction template matches (reversible families). 3. Timing: product enumeration for a 10-atom reactant < 5s (record). Mismatches in product sets are HARD failures. Any family whose data format cannot be expressed yet: BLOCKED family (listed; job 06 proceeds around them).

## When the job is done

The final step's report (reports/job-05.md) has the gate result, the
job table row is `done` (or `blocked` with the cause), and the NEXT pointer
in STATUS.md targets job-06's first step (if there is a next job).
