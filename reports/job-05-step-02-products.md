# job-05/step-02 report: Product enumeration (generate_reactions + degeneracy)

## What was built

- **rmgpu/core/enumeration.py** (new, ~950 lines) - the product-enumeration
  orchestration, a port of RMG-Py's `generate_reactions` / `calculate_
  degeneracy` / `find_degenerate_reactions` machinery onto the step-01
  recipe engine. Holds:
  - `Family` - a light holder for one family's enumeration-relevant data
    (recipe, reverse recipe, own_reverse, reversible, allow_charged_species,
    electrons, product/reactant counts, template labels, a `.matcher` hook);
    `Family.from_reference(case)` builds it from the recorded reference.
  - `TemplateReaction` - reactants/products (rmgpu Molecule pieces),
    degeneracy, reversible, family, is_forward, electrons, template,
    duplicate. The pieces keep their atom labels/IDs (the degeneracy
    machinery needs them).
  - `RecordedMatcher` - replays RMG-Py's own recorded applications (the
    step-02 reference) through rmgpu's `apply_recipe`. It is the
    non-circular basis of the parity check: same labeled reactant graph +
    same recipe -> same product structure, and RMG's exact per-atom IDs flow
    into the products so the identical/isomorphic degeneracy distinction is
    reproduced. It also exposes `match_molecule(form, slot, branch)` (returns
    `[]`) as the hook that step-03's real group matcher implements.
  - `generate_reactions(...)` - the enumeration: resonance expansion,
    deduplication, the degeneracy bookkeeping, the RMG products filter,
    per-raw-reaction template attachment (from the reference's
    `raw_templates`, RMG raw order).
  - `find_degenerate_reactions(...)` - ported **line-for-line** from RMG
    `common.find_degenerate_reactions`: the `identical` check uses
    `is_identical(strict=False)` (element + connectivity + atom-ID sets; bond
    order / radical / charge ignored), and isomorphic-but-different-template
    products are kept as **separate duplicate reactions** (RMG's `continue`
    on a different template).
  - `reduce_same_reactant_degeneracy` - RMG's Bishop-Laidler same-reactant
    reduction (2 reactants halve, 3 reactants /6).
  - `calculate_degeneracy(family, reaction, resonance=True)` - re-enumerates
    the reaction's reactants (all resonance forms + bimolecular swap
    branches), filters to the reaction's own products, collapses with the
    reaction's own template + same-reactant count, returns the degeneracy.
    A mismatch raises `DegeneracyError` (RMG's `KineticsError`).
- **rmgpu/core/recipe.py** (extended) - three changes, all in service of the
  benzene+H (cyclohexadienyl radical) case where RDKit's aromaticity
  kekulizer cannot resolve the product:
  - `CHANGE_BOND` now keeps RMG's **fractional** benzene-bond orders (1.5 -1
    -> 0.5, 1.5 +1 -> 2.5) instead of snapping to a whole number. RDKit has
    no native 0.5/2.5 bond type, so the true order is carried in the bond's
    `'order'` property and read back via the new `_bond_order()` /
    `_set_bond_order()` helpers.
  - A **DOF/valence kekulizer** (`_dof_kekulize`, `_AromaticRing`,
    `_AromaticBond`) ported from `rmgpy/molecule/kekulize.pyx` - it resolves
    1.5/0.5/2.5 bonds to S/D from the atoms' valences and has no aromaticity
    prerequisite.
  - `_all_six_rings` is now **connectivity-based** (a port of RMG
    `get_all_cycles_of_size(6)`) instead of RDKit's `GetRingInfo().AtomRings()`
    - RDKit's ring perception requires a valid (kekulized) molecule, which
    the semi-kekulized benzene+H product is not, so it returned no rings.
  - `_kekulize_piece` tries RDKit's kekulizer on a throwaway copy, then a
    residue check (any fractional bond left) routes to the DOF kekulizer
    (RMG's kekulizer either fails outright or reports success while leaving a
    non-aromatic ring partially resolved).
- **gates/baselines/job05/step02_products_reference.json** (new) - the
  recorded RMG-Py ground truth for the 30 cases: for each case the family
  metadata (recipe, reverse_recipe, own_reverse, ...), the labeled reactant
  applications exactly as RMG applied them (with RMG's per-atom IDs), the
  raw per-application templates (`raw_templates`, RMG raw order), the product
  structures (RMG adjacency lists) + degeneracy, the same-reactant flag, and
  RMG's own `calc_degeneracy` value per reaction.
- **scripts/record_job05_step02_reference.py** (new) - runs RMG-Py's
  `KineticsFamily.generate_reactions` / `calculate_degeneracy` (rmg_env,
  testing_database families) on the 30 cases and records the reference.
  Non-circular: uses only rmgpy.
- **scripts/capture_step02_app_products.py** (new) - records RMG's
  per-application products (with exact atom IDs) as a debug aid
  (supplements, does not replace, the reference).
- **scripts/_verify_step02.py** (new) - the 30-case parity check
  (product SMILES-sets + degeneracies, rmgpu replay vs the recorded
  reference).
- **tests/test_product_enum.py** (new, 35 tests) - the 30-case product-set +
  degeneracy parity, the reaction-count parity, and targeted tests for the
  three degeneracy/kekulize bugs (Bug A acetyl strict=False, Bug C benzene+H
  Kekule form).

## Checks run (real results)

- `/home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest
  tests/test_product_enum.py -q` -> **35 passed in 1.6s** (0 failed).
- 30-case reference parity: `.../python scripts/_verify_step02.py`
  -> **30 pass, 0 fail (of 30)**.
- `calculate_degeneracy` parity vs the recorded RMG `calc_degeneracy`
  (46 reactions across the 30 cases) -> **46 ok, 0 bad**.
- Full suite `.../python -m pytest tests/ -q` -> **581 passed, 1 failed**.
  The single failure (`test_ml_base.py::test_single_item_prediction_not_
  dropped`) is **pre-existing**: it fails identically on the clean tree
  (`git stash -u` then re-run -> same failure; `git stash pop`). It is the
  same-process checkpoint flake documented in the job-04 step-04/05 entries,
  unrelated to this step (ML base, not recipe/enumeration).

## Reference reads beyond the step's list

- `RMG-Py/rmgpy/data/kinetics/common.py` - `find_degenerate_reactions`,
  `reduce_same_reactant_degeneracy`, `check_for_same_reactants`,
  `same_species_lists` (the exact porting source for the degeneracy
  machinery).
- `RMG-Py/rmgpy/molecule/kekulize.pyx` - the DOF/valence kekulizer
  (`kekulize`, `AromaticRing`, `AromaticBond`, `prioritize_rings`,
  `prioritize_bonds`).
- `RMG-Py/rmgpy/molecule/graph.pyx` - `get_all_cycles_of_size(6)` (Fan,
  Panaye, Doucet, Barbu) for the connectivity-based ring enumeration.
- `RMG-Py/rmgpy/molecule/molecule.py` - `Bond._change_bond` /
  `increment_order` / `decrement_order` / `is_benzene`, `Atom.update_charge`
  / `get_total_bond_order`, `Molecule.kekulize`.
- `RMG-Py/rmgpy/molecule/element.py` - `PeriodicSystem.valences` /
  `valence_electrons` (the DOF analysis constants).
- `RMG-Py/rmgpy/data/kinetics/family.py` - `apply_recipe` (the `validAromatic`
  flag + kekulize trigger) and `R_Addition_MultipleBond` recipe (verified the
  `CHANGE_BOND *1 -1 *2` action produces a 0.5 bond in RMG, confirmed by
  running the recipe in rmg_env).
- The `rmg_env` python (RMG-Py checkout) to run RMG-Py directly for ground
  truth (the rmgpu env has no rmgpy).

## Deviations from this step file

1. **Kekulization is now RMG's DOF/valence resolver, not RDKit's
   aromaticity kekulizer.** The step-01 note said "RDKit's kekulizer returns
   a valid Kekule form, isomorphic but not string-identical." For the
   benzene+H case RDKit's kekulizer cannot resolve the product at all (the
   cyclohexadienyl radical has an sp3 CH2, a radical C, and a 0.5-order bond
   remnant; RDKit cannot ring-perceive it). The faithful RMG algorithm (the
   DOF/valence kekulizer) **was** ported and used as the fallback, so the
   product now matches RMG's exactly (canonical SMILES equal), not just
   isomorphically. This is a superset of the step-01 deviation, not a new
   one: aromatic products that RDKit *can* kekulize still go through RDKit
   first.
2. **`generate_reactions` signature.** The step file specifies
   `generate_reactions(reactants, products=None, prod_resonance=True,
   delete_labels=True, relabel_atoms=True)`. The implemented signature also
   takes `family` and `matcher` (required - the step-01 engine is a per-family
   recipe and there is no template yet; the `family` + `matcher` are the
   step-04 loader / step-03 group matcher respectively, and the `Recorded
   Matcher` is this step's stand-in for the latter). `template` is an extra
   kwarg for the degeneracy filter. This is consistent with the step-01
   deviation that "product_num is caller-resolved."
3. **The parity is against a *recorded* RMG-Py reference, not a live RMG-Py
   subprocess.** The reference (`step02_products_reference.json`) was recorded
   once by `record_job05_step02_reference.py` from the real RMG-Py
   (testing_database families) and is committed as the ground truth; the
   tests and `_verify_step02.py` compare the rmgpu replay against that file.
   This is the same non-circular-baseline pattern the repo already uses
   (job-01 round-trips, job-04 reference predictions, step-01's
   `step01_apply_recipe_reference.json`).

## What the next step (step-03 templates) should know first

- **The group matcher interface is the contract.** `generate_reactions`
  dispatches on `hasattr(matcher, 'yield_applications')`: the
  `RecordedMatcher` (this step) replays, and the step-03 **group matcher**
  implements `match_molecule(form, slot, branch)` -> a list of labelings
  (`{atom_index: label}`) of `form` against template reactant `slot` for
  reactant order `branch` (`'ab'` = A+B/template order, `'ba'` = the swapped
  B+A order). `RecordedMatcher.match_molecule` returns `[]`. The fresh
  (`_enumerate_fresh`) path already consumes that interface correctly
  (resonance expansion + `assign_fresh_ids` + both bimolecular branches +
  the not-isomorphic-swap guard) - step-03 only needs to supply real
  labelings.
- **`generate_reactions` copies the matcher's structures** (`m.copy(
  clear_labels=False)`) before applying, because `apply_recipe` /
  `clear_labeled_atoms` mutate them in place and the matcher must stay
  reusable across re-runs (e.g. `calculate_degeneracy` re-runs the
  enumeration). A real step-03 matcher may safely share Molecule objects.
- **Per-raw-reaction templates come from `raw_templates`** (RMG raw order) in
  the replay path and are essential: they are what keep isomorphic-but-
  different-template products as separate duplicate reactions instead of
  combining their degeneracy. The fresh path sets the family-level
  `template_labels` fallback where a template is still unset.
- **The DOF kekulizer reads bond orders via the bond `'order'` property**
  (fractional 0.5/2.5 orders RDKit cannot represent natively are stored there
  by `CHANGE_BOND`). Any step that builds or mutates a ring and then
  kekulizes must go through `_bond_order`/`_set_bond_order`, not raw
  `GetBondTypeAsDouble`.
- **`product_num` is still caller-resolved** (step-04 family loader will
  supply the template's product count). `apply_recipe`'s product-count check
  is skipped when `product_num` is None.
