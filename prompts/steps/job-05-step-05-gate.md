# job-05/step-05: Job-05 gate (product enumeration parity)

Job: job-05 - Reaction recipe DSL + product enumeration
Prereq: jobs 01-02 done (Molecule layer; family-definition storage decision from job-02 step 4)
This step is part of that job. The job's overall goal:
Port RMG's reaction GENERATION machinery: the recipe DSL (atom- labeled bond operations) that, given a reaction template + a reactant structure, enumerates all valid product structures with degeneracy. Custom RMG IP (NOT SMARTS reactions - PLAN.md 4), ported faithfully - the engine that turns "a C=C exists" into concrete isomer products. Deliverables: rmgpu/core/recipe.py (the engine), rmgpu/core/template.py (template matching), rmgpu/core/family.py (loader + facade), plus the group matcher in rmgpu/molecule/group.py (partial port).
The job's gate (run by the job's final step):
gates/gate_05.py over a fixed set of (family, reactants) cases (every family in the 'default' set that appears in the c3h4/superminimal mechanisms + the RMG-Py test fixtures): 1. Product enumeration parity: for each case, RMG-Py's family.get_products/apply vs rmgpu's: the SETS of products (canonical SMILES) equal, and the degeneracy per product matches EXACTLY. Report: cases, pass/fail, mismatches listed. 2. Reverse: for enumerated products, the reverse-reaction template matches (reversible families). 3. Timing: product enumeration for a 10-atom reactant < 5s (record). Mismatches in product sets are HARD failures. Any family whose data format cannot be expressed yet: BLOCKED family (listed; job 06 proceeds around them).

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

- Previous step (already done, its code is in the tree): job-05/step-04-families
- This step: job-05/step-05-gate
- Next step (do NOT start it): the job gate for job-05 (its step file)

## Goal

Run the job-05 gate: the full (family, reactants) case set, product
sets + degeneracies vs RMG-Py, reverse matching, timing, the blocked
families. Close job 05.

## Reference to read (this step's budget)

  (no new reference reads)
  RMG-Py/test/regression/{superminimal,c3h4}/input.py + the example mechanisms
    (the case set: every family in the 'default' set that appears in those
    mechanisms)
  RMG-Py/tests/rmgpy/data/kinetics/ (the full test fixtures)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- gates/gate_05.py: builds the case set (default-set families in
  the c3h4/superminimal mechanisms + the RMG-Py fixtures), runs rmgpu
  generate_reactions vs RMG-Py's family generation for each, compares
  product SMILES-sets + per-product degeneracies (must match EXACTLY),
  checks reverse matching on reversible families, records timing (10-atom
  reactant enumeration < 5s sanity).
- reports/job-05.md: cases, pass/fail, every mismatch listed (case, family,
  the two product sets, the degeneracy diffs), the blocked-families list,
  timing.
- The full tests/ suite green.
- If a product-set mismatch has a common cause (one rule, one label
  mechanic), root-cause it and fix in recipe.py/template.py (the responsible
  modules) - then re-run; a fix here is cheaper than in job 06.

## Checks (must run and pass before you claim done)

  /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_05.py
    -> PASS (product sets + degeneracies exact) or RED with the mismatches
    listed
  pytest tests/ -q -> all pass

## Pitfalls

- Product-set mismatches are HARD failures (they break mechanism
  generation in job 06) - do not close the job with unexplained mismatches.
- Degeneracy mismatches are HARD failures too (they shift the mechanism) -
  the "close enough" tolerance is 0 on this gate.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-05/step-05: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-05/step-05 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-05-step-05-gate.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
