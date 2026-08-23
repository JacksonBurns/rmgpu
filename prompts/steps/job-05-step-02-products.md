# job-05/step-02: Product enumeration (generate_reactions)

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

- Previous step (already done, its code is in the tree): job-05/step-01-engine
- This step: job-05/step-02-products
- Next step (do NOT start it): job-05/step-03-templates

## Goal

Port the product-enumeration orchestration: from a recipe applied to
reactants, generate the full product set (with resonance, deduplication,
degeneracy). This is what "a C=C exists -> concrete isomer products" means.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/data/kinetics/family.py lines ~950-1150
    (generate_product_template) and ~1765-2050 (generate_reactions,
    _generate_reactions, calculate_degeneracy, the resonance handling in
    product generation)
  (step 1's recipe.py; job-01 resonance for prod_resonance)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/core/recipe.py (extend):
    `generate_reactions(reactants, products=None, prod_resonance=True,
    delete_labels=True, relabel_atoms=True) -> (products, degeneracies)` -
    the full enumeration per RMG (step 1's apply as the core; this step adds
    the orchestration: product template generation, resonance expansion,
    dedup, the degeneracy bookkeeping).
    `calculate_degeneracy(reaction, resonance=True)` - port EXACTLY (counts
    equivalent applications; must match RMG-Py bit-for-bit on the test set
    or the mechanism diverges).
- tests/test_product_enum.py: on a 30-case set (families from the
  superminimal/c3h4 mechanisms - H_ABstraction, R_Addition, Intra_H_
  Abstraction, etc. x representative reactants): product SMILES-sets +
  degeneracies match RMG-Py (reference script -> gates/baselines/
  products/). A subset here; the gate (step 4) runs the full set.

## Checks (must run and pass before you claim done)

  pytest tests/test_product_enum.py -q -> all pass
  30-case reference: product sets + degeneracies match (diff recorded)

## Pitfalls

- Degeneracy is the subtle part (equivalent applications, stereo,
  duplicate labelings) - a wrong count silently shifts the mechanism
  (job 06's parity gate will catch it; better to catch it here).
- prod_resonance: the products get their resonance sets (job-01
  resonance) - RMG's dedup is over the resonance-expanded set; port that
  order of operations exactly.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-05/step-02: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-05/step-02 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-05-step-02-products.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
