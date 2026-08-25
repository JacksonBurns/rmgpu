# job-05/step-04: Family loader + KineticsFamilies facade

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

- Previous step (already done, its code is in the tree): job-05/step-03-templates
- This step: job-05/step-04-families
- Next step (do NOT start it): job-05/step-05-gate

## Goal

Load family definitions from the DB (per job-02 step-4's documented
strategy) and build the KineticsFamilies facade: the object the core loop
(job 06) asks for families, templates, and matching.

## Reference to read (this step's budget)

  (job-02 step-4's family-storage decision - the report)
  RMG-Py/rmgpy/data/kinetics/family.py lines ~150-300 (the Family/FamilyDict
    class: get_family, the families list, how families register templates +
    reverse families)
  RMG-database/input/kinetics/families/ (the family.py files: 2-3 read in
    full - R_H_Abstraction, H_ABstraction - to pin the data format the
    loader parses: recipes, templates, rate rules as data)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/core/family.py:
    `Family`: name, recipes (step 1's ReactionRecipe objects), templates
    (step 3's groups), rate rules (loaded as DATA - used only for
    depository/reverse-rate purposes in job 08; the rate-rule ESTIMATOR is
    deleted, PLAN.md 3).
    `KineticsFamilies`: loads all families (or a named subset - the 'default'
    set from RMG-database/input/kinetics/families/'s family list file) from
    the source job-02 step 4 decided (rmgdb tables if stored there; else
    the controlled parse of the family.py files - the parser lives in
    family.py, reusing step 1's recipe parser).
    API: `get_family(name)`, `families` (list), `match_reaction(reaction) ->
    (family, template)` (delegates to step 3).
    Reverse families: RMG's ownReverse/reverse-family bookkeeping - a
    family's reverse is how the core loop knows a reaction is reversible
    (job 06 uses it).
- tests/test_families.py: load all 'default' families without error; counts
  (families, recipes per family, templates per family) vs RMG-Py's load
  (reference script); match_reaction on 20 reactions agrees with RMG-Py;
  the blocked-families list (any family whose data format the loader cannot
  express - each with the exact construct + a note for the gate).

## Checks (must run and pass before you claim done)

  pytest tests/test_families.py -q -> all pass
  all 'default' families load; counts match RMG-Py (or the blocked list is
    non-empty and every blocked family is documented)

## Pitfalls

- The blocked-families list is load-bearing for job 06 (it proceeds
  around them) - an empty-but-wrong list is worse than an honest one.
- Rate rules are DATA here (storage) - if any code starts COMPUTING from
  them (estimating a rate from rule averages), stop: that is the deleted
  estimator.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-05/step-04: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-05/step-04 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-05-step-04-families.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
