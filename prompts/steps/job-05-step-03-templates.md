# job-05/step-03: Template matching + group matcher

Job: job-05 - Reaction recipe DSL + product enumeration
Prereq: jobs 01-02 done (Molecule layer; family-definition storage decision from job-02 step 4)
This step is part of that job. The job's overall goal:
Port RMG's reaction GENERATION machinery: the recipe DSL (atom- labeled bond operations) that, given a reaction template + a reactant structure, enumerates all valid product structures with degeneracy. Custom RMG IP (NOT SMARTS reactions - PLAN.md 4), ported faithfully - the engine that turns "a C=C exists" into concrete isomer products. Deliverables: rmgpu/core/recipe.py (the engine), rmgpu/core/template.py (template matching), rmgpu/core/family.py (loader + facade), plus the group matcher in rmgpu/molecule/group.py (partial port).
The job's gate (run by the job's final step):
gates/gate_05.py over a fixed set of (family, reactants) cases (every family in the 'default' set that appears in the c3h4/superminimal mechanisms + the RMG-Py test fixtures): 1. Product enumeration parity: for each case, RMG-Py's family.get_products/apply vs rmgpu's: the SETS of products (canonical SMILES) equal, and the degeneracy per product matches EXACTLY. Report: cases, pass/fail, mismatches listed. 2. Reverse: for enumerated products, the reverse-reaction template matches (reversible families). 3. Timing: product enumeration for a 10-atom reactant < 5s (record). Mismatches in product sets are HARD failures. Any family whose data format cannot be expressed yet: BLOCKED family (listed; job 06 proceeds around them).

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

- Previous step (already done, its code is in the tree): job-05/step-02-products
- This step: job-05/step-03-templates
- Next step (do NOT start it): job-05/step-04-families

## Goal

Port template matching: given a family's templates (groups of
reactant patterns) and a concrete reaction's molecules, find the matching
labelings. Includes the group matcher (RMG group semantics - port, do not
force RDKit SMARTS where the semantics differ).

## Reference to read (this step's budget)

  RMG-Py/rmgpy/molecule/group.py  (3118 - read the Group class +
    the matcher (match/match_function) + the constructs family templates
    actually use; port the matcher, delegate pure graph ops to RDKit where
    the semantics agree - document each delegation decision)
  RMG-Py/rmgpy/data/kinetics/family.py lines ~2663-2770 (get_reaction_
    template_labels, retrieve_template, get_labeled_reactants_and_products -
    the matching entry points)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/molecule/group.py: the group matcher - RMG group semantics
  (the group definitions are SMARTS-like with RMG-specific constructs; port
  the constructs family templates actually use - inventory them from 3-4
  families' template data; document what is ported vs delegated to RDKit
  SMARTS and why).
- rmgpu/core/template.py:
    `match(family, reaction) -> template_labels` - given a family's
    templates (reactant groups) and a concrete reaction (r     reactant mol, product mol), find the matching template labelings.
    `match(family, reaction) -> (template_labels, template)` - used by the
    core loop (job 06) to know which family produced a reaction.
- tests/test_template_match.py: for 40 (family, reaction) pairs (from the
  product-enumeration set: the reactions that were generated must match
  their family's template - a round-trip consistency check) + 10 negative
  cases (wrong family must NOT match): match agrees with RMG-Py (reference
  script).
- The group-construct inventory (which RMG-specific constructs appear in
  real family templates) - a short table in the report.

## Checks (must run and pass before you claim done)

  pytest tests/test_template_match.py -q -> all pass
  round-trip: every generated reaction from step 2's set matches its
    family's template (rmgpu) and the match agrees with RMG-Py

## Pitfalls

- Group matching is the other "port the semantics" surface (after
  recipes) - the delegation decisions (what goes to RDKit SMARTS) must be
  per-construct and documented; a blanket "use SMARTS" will drift on the
  RMG-specific constructs.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-05/step-03: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-05/step-03 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-05-step-03-templates.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
