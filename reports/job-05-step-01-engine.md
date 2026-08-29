# job-05/step-01 report: ReactionRecipe engine (apply_recipe + labels)

## What was built

- **rmgpu/core/recipe.py** (new, ~900 lines) - the ReactionRecipe engine, a
  pure-Python port of RMG-Py's recipe machinery (rmgpy/data/kinetics/family.py
  ReactionRecipe._apply + KineticsFamily.apply_recipe) onto rmgpu's RDKit-based
  Molecule. Holds:
  - `ReactionRecipe` (parse/hold actions via `from_data`, `get_reverse`),
  - `apply_recipe()` - merge reactants, re-aromatize, apply the recipe,
    kekulize invalidated aromatic rings, relabel self-reverse families on the
    MERGED structure (before the split), split into pieces, update product
    lone-pairs/charges (RMG update()), net-charge check, piece ordering,
  - the validity rules verbatim (bond order bounds, no-duplicate/no-missing
    bonds, radical/charge/lone-pair bookkeeping with RMG's
    update_charge-after-pair-action coupling),
  - label helpers for step 02 (`label_atoms`, `label_atoms_with_lone_pairs`,
    `clear_labeled_atoms`, `label_fingerprint`),
  - the reversible-family relabel table (RMG's hardcoded swaps + the
    reverse_map fallback).
- **tests/test_recipe_engine.py** (new, 27 tests) - the RMG-Py fixture cases
  (adjacency lists copied from RMG-Py's familyTest.py, attribution: MIT
  license, RMG-Py repo) run through the engine and compared against the
  recorded RMG-Py products (piece count, atom count, net charge, per-label
  structural fingerprint), plus recipe-object and validity-rule unit tests.
- **scripts/record_job05_step01_reference.py** (new) - runs RMG-Py's
  `KineticsFamily.apply_recipe` (rmg_env, RMG-Py testing_database families)
  on the exact fixtures and records the ground-truth products + each family's
  recipe + metadata (own_reverse, reverse_map, effective product/reactant
  counts, electrons) to gates/baselines/job05/
  step01_apply_recipe_reference.json. Non-circular: uses only rmgpy.
- **gates/baselines/job05/step01_apply_recipe_reference.json** (new) - the
  recorded ground truth (9 gas-phase apply_recipe cases + 2 surface cases +
  recipe action lists).
- **rmgpu/molecule/molecule.py** (extended, minimal) - `from_adjacency_list`
  now stores the RMG p-column (lone pairs) verbatim as the atom's 'lp'
  property (the recipe's PAIR bookkeeping and RMG's update_charge recompute
  need RMG's stored value, which for charged atoms differs from the
  neutral-formula default); `is_cyclic()` now sanitizes a throwaway copy when
  the stored Mol's ring info is uninitialized (structures built by the recipe
  engine are not sanitized).
- **rmgpu/molecule/adjlist.py** (extended, minimal) - `get_atoms_info` prefers
  the stored 'lp' property over the neutral-formula derivation, so the
  p-column round-trips verbatim.

## Checks run (real results)

- `rmgpu/bin/python -m pytest tests/test_recipe_engine.py -q`
  -> **27 passed** (0 failed).
- 9-case apply_recipe parity smoke (engine vs the recorded RMG-Py ground
  truth; piece count + atom count + net charge + label fingerprint per piece,
  in RMG's product order):
  - H_Abstraction: OK
  - R_Addition_MultipleBond_benzene: OK
  - intra_H_migration: OK
  - Intra_ene_reaction: OK
  - 6_membered_central_C-C_shift: OK
  - 1,2_shiftC: OK
  - Intra_R_Add_Exo_scission: OK
  - intra_substitutionS_isomerization: OK
  - R_Addition_COm: OK
- Full suite `rmgpu/bin/python -m pytest tests/ -q` -> **545 passed, 2
  failed**. The 2 failures are `tests/test_ml_base.py::test_kinetics_matches_reference`
  and `::test_kinetics_deterministic` - the documented order-dependent
  checkpoint-drift flakiness (kinetics checkpoint results drift ~1.6e-2
  relative on Ea when run after other kinetics tests in the same process;
  both pass in isolation: `pytest tests/test_ml_base.py -q` -> 10 passed).
  Pre-existing, unrelated to this step's files.

## Reference reads beyond the step's list

- rmgpy/molecule/molecule.py - Atom.apply_action, increment/decrement
  radical/charge/lone_pairs, update_charge, Molecule.update / update_lone_pairs
  (the exact charge/lone-pair coupling the PAIR actions and the product
  update step rely on).
- rmgpy/molecule/molecule.py Bond.set_order_str / _change_bond / is_benzene
  (bond-order bookkeeping).
- rmgpy/molecule/kekulize.pyx (skimmed; the port uses RDKit's kekulizer
  instead - see deviations).
- rmgpy/data/kinetics/family.py apply_recipe lines 1339-1608 (the full
  function, not just the budgeted range, to confirm the relabel-before-split
  ordering and the product update/charge steps).

## Deviations (with cause)

1. **Kekulization via RDKit, not RMG's DOF analysis.** RMG kekulizes with its
   own DOF-based resolver; this port re-aromatizes ring bonds to 1.5 when
   merging and lets RDKit's `Chem.Kekulize` resolve them. The products are
   isomorphic to RMG's (same molecule), but the exact S/D bond placement can
   differ (e.g. 1,3,5- vs 1,4,6-triene), so a canonical SMILES string can
   differ for aromatic products. The step checks and the job-05 gate therefore
   compare structure (label fingerprints / atom counts / net charge /
   formula + radical count), not raw SMILES. Recorded here so the gate
   (step 05) compares structure, not string.
2. **`product_num` default semantics.** RMG resolves the expected product
   count from the template (`self.product_num or len(forward_template.products)`
   / `self.reactant_num or len(reverse_template.products)` for reverse). The
   rmgpu engine is a free function with no template, so the caller (step 02's
   product enumeration) must resolve and pass it; the reference records the
   effective counts. When `product_num=None` the count check is skipped (RMG
   always has a template, so it can't distinguish).
3. **Surface families out of scope.** The two
   Surface_Dissociation_Charge_Separation cases are recorded in the reference
   (for job-12) but the gas-phase engine cannot build 'X' site atoms, so they
   raise - a test documents this.

## What step-02 (product enumeration) should know first

- The engine's `apply_recipe` returns the product Molecules in RMG's order
  (piece carrying '*1' first of two; three pieces by lowest label), with
  labels already relabeled for self-reverse families. The product
  enumeration (generate_reactions) must supply the template-resolved
  `product_num`, `own_reverse`, `reverse_map`, `electrons`, and the family
  label - all available from the family loader (step 04) / group matcher
  (step 03).
- `label_fingerprint(molecule)` and `_ProductPiece.label_fingerprint()` give
  the order-independent label placement used to verify a product against the
  template's product labels (the degeneracy/dedup step).
- `label_atoms(structures, maps)` / `label_atoms_with_lone_pairs(...)` seed
  the labels the engine expects on the reactants; the 'lp' property is
  authoritative for PAIR bookkeeping.
- Aromatics: product SMILES may differ from RMG's exact kekulization
  (isomorphic only). Structure comparison, not string comparison, is the
  parity criterion (gate 05 must use isomorphism / canonical SMILES of the
  kekulized molecule, and should dedup products by that).
- The reference JSON is the step-02 ground truth for the per-case recipes +
  metadata; extend it per family as the group matcher lands.
