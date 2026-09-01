# job-05/step-04: Family loader + KineticsFamilies facade

## What was built

- **`rmgpu/core/family.py`** (new, ~730 lines) - the family loader + the
  `KineticsFamilies` facade the core loop (job 06) asks for families,
  templates, and matching.
  - `Family.from_files(name, fam_dir, kinetics_db)`: builds one family's data
    - name/identity, the recipe (`ReactionRecipe`) + reverse recipe (step 1),
      the group tree (`entries`/`top`/`forward_template`, step-3 shapes), the
      rate rules (parsed as DATA), the forbidden structures (as DATA), and the
      reverse-family bookkeeping (`reverse`/`reversible`/`own_reverse`/
      `reverse_map`/`reactant_num`/`product_num_forward`).
  - Controlled parse of `<fam_dir>/<name>/groups.py` (`_parse_groups_file`) -
    the same mechanism rmgdb's build and RMG-Py `Database.load` use (exec in a
    stub namespace: `entry`/`template`/`recipe`/`tree`/`forbidden`). No DSL
    re-invented.
  - `rmgdb` cross-check (`_cross_check_rmgdb`): the parse's
    template/recipe/reversible/reverse_map/reactant_num/product_num/
    auto_generated + every group's label and adjacency list must agree with the
    `kinetics_families_table` / `kinetics_family_groups_table` rows. Any
    disagreement is a `FamilyLoadError`. Descriptions (`short_desc`/`long_desc`)
    are captured from the cross-check so `load` does not re-parse.
  - `_load_tree`: RMG-Py `Database._load_tree` ported (parent =
    `parents[level-1]`, top = no parent, first-appearance order). Comment
    stripping matches RMG `remove_comment_from_line` EXACTLY (only a `//`
    comment is cut - `#` is a legitimate character in tree labels, e.g.
    `C_rad/H2/Cs\H2\Cs|Cs#O`).
  - `_parse_rules_file`: the rate rules as plain DATA (kinetics kind + raw
    constructor args, incl. nested `RateUncertainty`), captured by stubs, never
    evaluated (the rate-rule estimator is deleted, PLAN 3/4).
  - `KineticsFamilies`: `.load('default'|'all'|[names])`, `.get_family(name)`,
    `.families`, `.match_reaction(reaction) -> (family, template_labels)`
    (delegates to step 3 `template.match`), `.get_families_of_reaction`,
    `.blocked` (`{name: {reason, construct, note}}`).

- **`rmgpu/molecule/group.py`** - added `Group.split()` (connected-component
  split of a group's explicit graph, atom order preserved) - RMG's
  `Group.split` used for the single-template-reactant split
  (R_Recombination / Birad_recombination's `Root`: two radical atoms -> 2
  reactants).

- **`tests/test_families.py`** (new, 7 tests) - all 51 'default' families load
  with no error + empty blocked list; counts (families, recipes per family,
  templates per family) agree with the recorded reference; reverse-family
  bookkeeping agrees; `match_reaction` on the 20 reactions agrees with RMG-Py
  (family + labels, modulo one documented aromatic exception); the
  blocked-families mechanism records `reason`/`construct`/`note`.

- **`gates/baselines/job05/step04_families_reference.json`** (new) +
  **`scripts/record_job05_step04_reference.py`** (new) - the recorded RMG-Py
  reference (51 families' enumeration fields + 20 match_reaction verdicts),
  run in `rmg_env`.

## Checks run (real results)

- `pytest tests/test_families.py -q` -> **7 passed** (load, counts, reverse
  bookkeeping, match_reaction family agreement, match_reaction label agreement,
  blocked mechanism, missing-file error).
- `pytest tests/test_families.py tests/test_template_match.py
  tests/test_recipe_engine.py tests/test_product_enum.py -q` -> **75 passed**
  (no job-05 regressions).
- `pytest tests/ -q` -> **595 passed, 68 warnings in 19.6s** (full suite green).
- `KineticsFamilies().load('default')` -> 51 families, 0 blocked, 0.5s; rules
  loaded 51/51 (0 `rules_error`); 26 families carry forbidden structures.
- `match_reaction` on the 20 recorded reactions -> 20/20 attributed to the
  family RMG-Py says matches; 1.0s total (the 10-atom reactant timing gate is
  a job-05/gate concern, not this step's).
- Count parity vs reference: 0 field mismatches across all 51 families
  (template reactant/product labels, recipe action count + content, reverse
  recipe content, group entry count, own_reverse, reversible, reverse_map,
  allow_charged_species, electrons, stored reactant_num, effective
  reactant_num, effective product_num).

## Reference reads beyond the listed budget

- RMG-Py `rmgpy/data/base.py` `_load_tree` (497-553) + `load` (204-264) and
  `remove_comment_from_line` (1281) - to port the tree parse and confirm the
  `//`-only comment rule.
- RMG-Py `rmgpy/data/kinetics/family.py` `load` (635-700) and
  `get_labeled_reactants_and_products` (2689-2772) - to pin the
  `reactant_num`/`own_reverse`/`reversible`/reverse semantics and the
  `auto_generated` count guard.
- RMG-Py `rmgpy/data/kinetics/groups.py` `get_reaction_template` (111-201) -
  to confirm the descent starts from `groups.top` (all tree top nodes), which
  is why a unimolecular family's template labels include the end-root
  children (e.g. intra_H_migration -> 3 labels).

These are the "direct dependencies hit while implementing" the loader; they
define the exact load semantics the loader mirrors.

## Deviations from the step file

- The step lists the family files at
  `RMG-database/input/kinetics/families/` and the 'default' set "from
  `families/`' family list file". The 'default' set is the `default` variable
  in `families/recommended.py` (exec'd, the same source of truth RMG-Py's
  `load_recommended_families` uses); 51 families.
- The reference `scripts/record_job05_step04_reference.py` records the
  effective reactant/product counts as RMG's `family_meta` resolves them
  (`item.split()` on the one-group case). `Family` therefore carries BOTH the
  stored `reactantNum` (on `reactant_num`, the value the matcher's count guard
  and RMG's `auto_generated` guard read) and the separate
  `num_template_reactants_effective` (the split count, used for count parity
  only). This is what lets the facade agree with RMG-Py on the 20 reactions
  (see the Birad_recombination note below).

## Findings / what the next step should know first

1. **`fam.top` = the group tree's top nodes (all of them), NOT just the
   template reactant slots.** The step-3 matcher descends from `fam.top` and
   `match()` indexes `fam.top[slot]` for the per-slot match. For the 7
   mechanism families the top order matches the template reactant order; for
   16 of the 51 default families the tree has extra top nodes (unimolecular
   end-roots) - `match_reaction` on the 20 reactions is unaffected because
   those families' extra tops carry no recipe labels on the matched reactant.
   Job 06 / the gate should keep `fam.top` = tree tops (this is the
   RMG-faithful representation; `fam.forward_template` holds the template
   reactant slots).

2. **The effective reactant count vs the stored flag is load-bearing for
   `match_reaction` parity.** `Birad_recombination` has stored
   `reactantNum=1` + `autoGenerated=1`; RMG's `auto_generated` count guard
   (`auto_generated and reactant_num != len(reactants)`) therefore REJECTS the
   2-radical recombination reactions (OH+OH, CH3+OH) - they belong to
   `R_Recombination` (stored `reactantNum=2`). `Family.reactant_num` holds the
   stored flag so the matcher's `len(reactants) != fam.reactant_num` guard
   reproduces that rejection exactly (verified: real RMG-Py
   `get_labeled_reactants_and_products` returns `(None, None)` for
   Birad_recombination on OH+OH for this reason; the products DO generate to
   HOOH, so the rejection is the guard, not the recipe). Setting `reactant_num`
   to the split count instead would make the matcher accept those reactions
   into Birad_recombination (a false positive vs RMG-Py). `match_reaction`
   attribution: first family in load order (sorted name) whose `match()`
   returns labels - `Birad_recombination` sorts before `R_Recombination`, so
   the guard is what keeps OH+OH attributed to `R_Recombination`.

3. **The 20-reaction label parity has exactly one documented exception class**
   (2 cases): benzene substrates. rmgpu stores molecules Kekulized (Cd
   carbons), RMG-Py aromatic (Cb carbons); the match verdict + reactant
   labeling agree, only the most-specific descent leaf differs
   (`H_Abstraction|4|0`: `Cd/H/Cd` vs `Cb_H`; `R_Addition_MultipleBond|19|0`:
   `Cds-CdH_Cds-CdH` vs `Cb-H_Cb-H`). Same documented exception as step 03.

4. **rmgdb's tree table is incomplete - the file parse is primary (verified).**
   `kinetics_family_groups_tree_table` is missing parent/child edges for 21 of
   the 51 default families (H_Abstraction 0/534, R_Addition_MultipleBond
   0/1211, intra_H_migration 0/310, several others 0). rmgdb `build.py`
   derives edges with `tree_str_to_pairs.sketchy_conversion`, which crashes on
   those `tree()` strings and swallows the exception. The loader therefore
   reconstructs the hierarchy from the `tree()` string (RMG `_load_tree`
   semantics) and uses rmgdb only to cross-check flat data (family row + group
   adjacency lists), which match exactly.

5. **Rate rules + forbidden structures are DATA** (rules kind + raw args;
   forbidden label + group). Nothing in `family.py` computes a rate from the
   rules - that is the deleted estimator (PLAN 3/4). Job 08 (depository /
   reverse-rate) consumes them as stored.

6. **Blocked list is empty for the 'default' set** (51/51 load). The mechanism
   is validated (a family with no `groups.py` in a temp dir lands in
   `.blocked` with reason/construct/note). Job 06 proceeds around anything
   that ends up blocked; for 'default' there is nothing to proceed around.

Next step: job-05/step-05-gate (product enumeration parity gate over the fixed
(family, reactants) case set; `match_reaction` from this facade + the step-02
`generate_reactions` are the two halves it joins).
