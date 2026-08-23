# job-05/step-01: ReactionRecipe engine (apply_recipe + labels)

Job: job-05 - Reaction recipe DSL + product enumeration
Prereq: jobs 01-02 done (Molecule layer; family-definition storage decision from job-02 step 4)
This step is part of that job. The job's overall goal:
Port RMG's reaction GENERATION machinery: the recipe DSL (atom- labeled bond operations) that, given a reaction template + a reactant structure, enumerates all valid product structures with degeneracy. Custom RMG IP (NOT SMARTS reactions - PLAN.md 4), ported faithfully - the engine that turns "a C=C exists" into concrete isomer products. Deliverables: rmgpu/core/recipe.py (the engine), rmgpu/core/template.py (template matching), rmgpu/core/family.py (loader + facade), plus the group matcher in rmgpu/molecule/group.py (partial port).
The job's gate (run by the job's final step):
gates/gate_05.py over a fixed set of (family, reactants) cases (every family in the 'default' set that appears in the c3h4/superminimal mechanisms + the RMG-Py test fixtures): 1. Product enumeration parity: for each case, RMG-Py's family.get_products/apply vs rmgpu's: the SETS of products (canonical SMILES) equal, and the degeneracy per product matches EXACTLY. Report: cases, pass/fail, mismatches listed. 2. Reverse: for enumerated products, the reverse-reaction template matches (reversible families). 3. Timing: product enumeration for a 10-atom reactant < 5s (record). Mismatches in product sets are HARD failures. Any family whose data format cannot be expressed yet: BLOCKED family (listed; job 06 proceeds around them).

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
- This step: job-05/step-01-engine
- Next step (do NOT start it): job-05/step-02-products

## Goal

Port the recipe ENGINE core: ReactionRecipe - holding the recipe
(bond operations over labeled atoms) and applying it to reactants. This is
the heart of RMG's generation machinery - port RMG's exact validity rules.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/data/kinetics/family.py lines ~301-790 (ReactionRecipe:
    class, load/load_template/load_recipe, the recipe action semantics -
    the bond-ops list with labeled atoms, e.g. R-H + X* -> R-X + H*)
  RMG-Py/rmgpy/data/kinetics/family.py lines ~1339-1690 (apply_recipe,
    _generate_product_structures' label mechanics, _create_reaction - read
    for the mechanics; the product-enumeration orchestration is step 2)
  RMG-Py/rmgpy/molecule/molecule.py (the label/bond-mutation APIs the recipe
    uses - rmgpu's equivalents from job 01; where job-01 lacks an API, add
    the minimum to rmgpu/molecule/ now - keep it small)
  RMG-Py/tests/rmgpy/data/kinetics/ (the recipe test fixtures - find + copy
    the relevant ones into rmgpu/tests/ with attribution)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/core/recipe.py:
    `ReactionRecipe`: parses/holds the recipe (the list of bond operations
    with labeled atoms) from the family.py data format (the format is
    data-ish Python - the job-02 step-4 decision names the loader; the
    recipe PARSER lives here), and `apply(reactants) -> (products,
    degeneracies)` enumerating all valid applications over atom labelings,
    with RMG's exact validity rules (valence checks via the molecule layer,
    bond-order bookkeeping, no-duplicate products, relabel_atoms semantics).
    Labeled-atom mechanics: port RMG's labeled-atom + _label_atoms semantics
    on the rmgpu Molecule (labels '1','2','*','*1', bond labels).
- tests/test_recipe_engine.py: the copied RMG-Py fixtures: a recipe applied
  to a reactant -> the labelings + the bond mutations match RMG-Py (a
  reference script in the report; the gate generalizes it). Focus: valid
  vs invalid applications (valence, bond order), the relabel behavior.

## Checks (must run and pass before you claim done)

  pytest tests/test_recipe_engine.py -q -> all pass

## Pitfalls

- This is the most "port RMG's exact semantics" job after pdep.
  When in doubt, RMG-Py's behavior on a given case is the spec - do not
  "improve" a rule.
- Do NOT port the rate-rule training code (depository -> rules, the BM
  tree, prune_tree, get_training_set) - deleted (PLAN.md 3). Only the
  recipe/template machinery.
- Where the recipe needs a Molecule API job 01 did not build, add the
  minimum to rmgpu/molecule/ (note it in the report) - do NOT fork the
  recipe's logic around it.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-05/step-01: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-05/step-01 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-05-step-01-engine.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
