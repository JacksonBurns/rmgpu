# job-05: Reaction recipe DSL + product enumeration

Read ORIENTATION.md (retain #2) and PLAN.md section 4 (retained: recipes). Prereqs:
jobs 01-02 (molecule layer + database/family loading).

## Goal

Port RMG's reaction *generation* machinery: the recipe DSL (atom-labeled bond
operations) that, given a reaction template + a reactant structure, enumerates all
valid product structures with degeneracy. This is custom RMG IP (NOT SMARTS
reactions - PLAN.md section 4) and must be ported faithfully; it is the engine that
turns "a C=C exists" into concrete isomer products.

## Reference (read - this is a big port; read the actual code)

  RMG-Py/rmgpy/data/kinetics/family.py  (245k) - focus on:
    ReactionRecipe (apply_recipe, _generate_product_structures, _get_degeneracy,
    label handling), TemplateReaction/TemplateGroup matching,
    calculate_degeneracy, _find_matching_atoms.
  RMG-Py/rmgpy/molecule/molecule.py - the label/bond-mutation APIs the recipe uses
    (use rmgpu's Molecule equivalents from job 01; where job-01 lacks an API, add it
    to job-01's modules now - that is allowed, keep it small).
  Family data: RMG-Py loads families from RMG-database/input/kinetics/families/<N>/
  (family.py + templates + rate rules). Job-02 documented how families are (or are
  not) stored in rmgdb - use that conclusion. The recipe DEFINITIONS (the bond-ops
  lists) are data; the recipe ENGINE is code we port.

## Deliverables

1. `rmgpu/core/recipe.py`
   - `ReactionRecipe`: parses/holds the recipe (the list of bond operations with
     labeled atoms, e.g. R-H + X* -> R-X + H*), and `apply(reactants) ->
     (products, degeneracies)` enumerating all valid applications over atom
     labelings, with RMG's exact validity rules (valence checks via the molecule
     layer, bond-order bookkeeping, no-duplicate products).
   - labeled-atom mechanics: port RMG's labeled-atom + `_label_atoms` semantics on
     the rmgpu Molecule (labels '1','2','*','*1', bond labels).
   - degeneracy counting: port `calculate_degeneracy` exactly (it counts equivalent
     applications; this must match RMG-Py bit-for-bit on the test set or the
     mechanism diverges).
2. `rmgpu/core/template.py`
   - reaction template matching: given a family's templates (groups of reactant
     patterns) and a concrete reaction's molecules, find matching labelings
     (group matching = substructure with RMG group semantics; RMG-Py's Group class
     in molecule/group.py is the reference - decide how much to port vs delegate to
     RDKit SMARTS; RMG groups are SMARTS-like but have RMG-specific constructs;
     port the group matcher, do NOT try to force RDKit SMARTS where semantics
     differ).
   - `match(family, reaction) -> template_labels` used by the core loop (job 06)
     to know which family produced a reaction.
3. `rmgpu/core/family.py` (loader + facade)
   - load family definitions from RMG-database (per job-02's documented strategy):
     recipes, templates, rate rules (rate rules loaded as DATA; they are used only
     for depository/reverse-rate purposes in job 08 - the rate-rule ESTIMATOR is
     deleted, PLAN.md 3).
   - `KineticsFamilies` aggregate: `get_family(name)`, `families` list,
     `match_reaction(reaction) -> (family, template)`.
4. Unit tests: port the recipe test cases from RMG-Py (RMG-Py/tests/rmgpy/data/
   kinetics/test_family.py and similar - find and copy the relevant fixtures with
   attribution).

## Gate (job 05) -> gates/gate_05.py, report reports/job-05.md

For a fixed set of (family, reactants) test cases (build from: every family in the
'default' set that appears in the c3h4/superminimal mechanisms + the RMG-Py test
fixtures):
  1. Product enumeration parity: for each case, run RMG-Py's family.apply/get
     products and rmgpu's; the SETS of products (canonical SMILES) must be equal,
     and the degeneracy per product must match exactly. Report: cases, pass/fail,
     any product-set or degeneracy mismatches (list them).
  2. Reverse: for products enumerated, the reverse-reaction template must also
     match (families that are reversible).
  3. Timing: product enumeration for a 10-atom reactant < 5s (sanity; record times).
  Mismatches in product sets are HARD failures for the job (they break mechanism
  generation). Document any family whose data format rmgdb/family-loading cannot
  yet express as a BLOCKED family (list them; job 06 can proceed around them).

## When done

STATUS.md + commit "job-05: reaction recipe DSL + product enumeration" + STOP.

## Pitfalls

- This is the most "port RMG's exact semantics" job after the pdep. The degeneracy
  and validity rules are subtle (stereochemistry, duplicate labeling, radical
  sites). When in doubt, RMG-Py's behavior on a given case is the spec.
- Do NOT port the rate-rule training code (depository -> rules, the BM tree) -
  deleted. Only the recipe/template machinery.
- Group matching (molecule/group.py, 135k) may need a partial port: port the
  matcher + the constructs actually used by family templates. Keep it in
  rmgpu/molecule/group.py (add to job-01's layer).
- If a family's recipe uses constructs rmgpu's Molecule doesn't support (rare
  atom types, unusual bonds), add the minimum Molecule support and note it.
