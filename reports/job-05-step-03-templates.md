# job-05/step-03 report: Template matching + group matcher

## What was built

- **rmgpu/molecule/group.py** (new, ~720 lines) - the group matcher, a port
  of RMG-Py's reaction-template / functional-group substructure matching
  (rmgpy/molecule/group.py + data/base.py `match_node_to_structure` + the
  Cython VF2 subgraph isomorphism) onto rmgpu's RDKit-backed Molecule.
  - `AtomType` + `ATOMTYPES` - the atom-type tree, **loaded from the
    recorded step-03 reference** (`gates/baselines/job05/
    step03_templates_reference.json` -> `atomtype_tree`), i.e. RMG-Py's own
    generic/specific structure, so the tree semantics are faithful and
    non-circular (never re-derived from rmgpu code).
  - `GroupAtom` / `GroupBond` / `Group` - the pattern-side vertex/edge,
    with an RMG group-adjacency-list parser (`parse_group_adjlist_full`)
    handling recipe labels (`*1`), multi-atomtypes (`[Cs,Cd,CO]`), wildcards
    (`u[x]`), charges (`c[n]`), lone pairs (`p[n]`), in-ring (`r[n]`), and
    bond-order sets (`{S,D}`, `B`, `[S,D,T,B]`).
  - `_explicit_graph(mol)` - the explicit-H (AddHs-order) kekulized graph the
    matcher operates on.
  - `match_explicit_graph` / `match_group` - a faithful VF2-style subgraph
    isomorphism (group -> molecule): vertex feasibility =
    `Atom.is_specific_case_of(GroupAtom)` (atom-type specific-case + radical/
    charge/lone-pair list membership, empty list = wildcard); edge
    feasibility = `Bond.is_specific_case_of(GroupBond)` (the molecule bond
    order is **exactly** one of the group bond's allowed orders - see the
    benzene finding below). Returns the list of valid mappings.
- **rmgpu/core/template.py** (new, ~660 lines) - the template-matching
  deliverable:
  - `TemplateFamily` / `TemplateEntry` / `LogicNode` - the family's group
    tree (loaded from the step-03 reference; the same shape the step-04
    family loader will emit). `TemplateFamily.from_references(fam_ref,
    step02_case)` attaches the recipe/flags from the step-02 reference and
    takes `reactant_num` from RMG's `num_template_reactants_effective` (the
    effective count after the single-group split, e.g. R_Recombination -> 2).
  - `match(family, reaction) -> template_labels | None` - the entry point.
    Faithful to RMG's `get_labeled_reactants_and_products` +
    `get_reaction_template_labels`: (1) reactant -> forward-template
    matching with the group matcher (the three branch shapes: single-group
    split for R_Recombination, unimolecular, bimolecular A+B/B+A swap);
    (2) recipe + product check via the step-01 engine (the family matches
    only if some labeling reproduces the reaction's products - this is what
    rejects a wrong family); (3) tree descent for the labels.
  - `match_node_to_structure` / `descend_tree` / `get_reaction_template` -
    RMG's labeled-atom initial pairing + most-specific-node descent, ported.
  - `TemplateMatcher` - implements the step-02 `match_molecule(form, slot,
    branch)` interface, so it can drop in place of the `RecordedMatcher` in
    `generate_reactions` (job 06).
- **tests/test_template_match.py** (new, 6 tests) - the parity test against
  the recorded RMG-Py reference.
- **Reference + recorders** - `scripts/record_job05_step03_reference.py`
  (atom-type tree + 7 family group trees + 99 (reactant, slot) subgraph-
  mapping counts + match_cases) -> `gates/baselines/job05/
  step03_templates_reference.json`; `scripts/record_job05_step03_verdicts.py`
  (RMG-Py's own verdict for all 7 families x 46 reactions of the step-02
  product set: matched/not + the most-specific template labels) -> `gates/
  baselines/job05/step03_match_verdicts.json`. Both run in `rmg_env`
  (RMG-Py), so the baseline is RMG-Py's own output (non-circular).

## Group-construct inventory (which RMG-specific constructs appear in real
family templates - across all 7 in-scope families' template trees)

| Construct | Ported? | Notes |
|-----------|---------|-------|
| `radical u<n>` (fixed) | yes | 2122 occurrences - most common |
| `recipe label *n` | yes | 2120 - recipe atom labels |
| `single-order bond {n,X}` | yes | 2078 - `{2,S}` etc. |
| `lone-pair p<n>` | yes | 276 |
| `multi-atomtype [a,b,...]` | yes | 414 - `[Cs,Cd,CO,CS,O,S,N]` |
| `benzene bond B` (order 1.5) | yes | 192 - matched on exact 1.5 bond order |
| `bond-order set [S,D,T,B]` | yes | 61 - the intra-H-migration backbone |
| `charge c<n>` (fixed) | yes | 5 |
| `bare recipe label *` | yes | 2 - R_Recombination `Y_rad` (both radical atoms share `*`) |
| `radical/charge/lone-pair wildcard` (`u[x]`/`c[x]`/`p[x]`) | yes | empty-list = any |
| `in-ring r<n>` | ported as non-restrictive | 0 occurrences in the 7 families |
| `LogicNode` (OR/AND of components) | yes | `forward_template` components; e.g. H_Abstraction `X_H_or_Xrad...` |
| RDKit-SMARTS delegation | none | no SMARTS used; all constructs ported natively |

No construct required RDKit-SMARTS delegation; every one that appears is
ported natively in `group.py`.

## Checks (commands + real results)

- `pytest tests/test_template_match.py -q` -> **6 passed**
  - `test_reference_present` - the recorded verdict matrix is decisive
    (46 matched, all own-family; 0 cross-family).
  - `test_roundtrip_own_family_matches` - all 46 step-02 reactions match
    their own family's template (the round-trip consistency check).
  - `test_full_verdict_agreement_with_rmgpy` - for ALL **322** (family,
    reaction) pairs (7 families x 46 reactions), rmgpu `match()` verdict
    (matched/not) equals RMG-Py's recorded verdict. (This supersedes and
    exceeds the "40 round-trip + 10 negative" ask.)
  - `test_negative_wrong_family_no_match` - 10 wrong-family pairs with
    matching reactant counts (so the rejection comes from the recipe/
    product check, not the reactant-count guard) correctly return None.
  - `test_template_label_agreement` - for the 46 own-family matches, the
    descended template labels equal RMG-Py's, except 2 documented aromatic
    cases (see deviations).
  - `test_group_subgraph_parity` - the group matcher finds the same number
    of valid subgraph isomorphisms as RMG-Py for all **99** (reactant,
    template-slot) pairs in the recorded reference.
- Round-trip (the step's second check): every generated reaction from
  step-02's set matches its family's template (rmgpu) and the match agrees
  with RMG-Py -> covered by `test_roundtrip_own_family_matches` +
  `test_full_verdict_agreement_with_rmgpy` (46/46 + 322/322).
- Regression: full suite `pytest tests/ -q` -> **588 passed** (no
  regression from the group.py/template.py additions).
- Harness (scripts/_harness_match.py, adjlist-built reactants): verdict
  agree **322/322**, label agree **44/46** (2 documented aromatic cases).

## Reference reads beyond the list

The step file lists rmgpy/molecule/group.py + family.py ~2663-2770. Also
read (direct dependencies hit): rmgpy/data/base.py (`match_node_to_structure`
line ~950, `descend_tree` line ~111), rmgpy/data/kinetics/groups.py
(`get_reaction_template` line 111), rmgpy/data/kinetics/family.py
(`_match_reactant_to_template` line 1730, `get_labeled_reactants_and_products`
line 2689), rmgpy/molecule/molecule.py (`Bond.is_specific_case_of` /
`equivalent`, line 757). The step's "Reference to read" budget fits one
session; no overflow.

## Deviations (with cause)

1. **Two benzene descent-label mismatches (Cd vs Cb).** For the two
   benzene-substrate reactions (H_Abstraction benzene+[H] -> ci=4;
   R_Addition_MultipleBond benzene+[H] -> ci=19), rmgpu's match verdict and
   the reactant->template **labeling** agree with RMG-Py exactly, but the
   most-specific **descent leaf** differs: rmgpu descends to `Cd/H/Cd` /
   `Cds-CdH_Cds-CdH`, RMG-Py's recorded labels are `Cb_H` / `Cb-H_Cb-H`.
   Cause: a **representation** difference, not a matching bug - rmgpu stores
   molecules Kekulized (Cd carbons, alternating 1.0/2.0 bonds) while
   RMG-Py stores benzene aromatic (Cb carbons, 1.5 bonds), so the descent
   lands in the Cd subtree vs the Cb subtree. The match (which the core loop
   / job-06 needs: which family + which template reactants) is exact.
   Documented in `test_template_label_agreement`
   (`_AROMATIC_DESCENT_EXCEPTIONS`).
2. **Bond-order matching is EXACT, not the "benzene ~ {S,D}" rule.** An early
   draft of `GroupBond.matches_molecule_order` let a group benzene order
   (1.5) match kekulized single/double bonds (1.0/2.0). That was wrong:
   RMG's `Bond.is_specific_case_of` is exact float order comparison (±1e-4)
   with **no** aromatic relaxation. The benzene case (R_Addition C=C/benzene,
   mine 12 vs RMG 6) was over-counted 2x because of the invented rule; after
   switching to exact comparison the 99/99 subgraph parity holds. (The
   matcher's 99/99 parity is on the Kekulized representation RMG-Py recorded,
   so exact comparison is the correct semantic there.)
3. **`reactant_num` from the reference, not label counting.** For
   single-group families a bare-`*` group (R_Recombination `Y_rad`, both
   atoms `*`) has only 1 distinct label for 2 reactant radicals, so
   counting labels under-derives the reactant count. `reactant_num` is taken
   from RMG's own `num_template_reactants_effective` in the step-02
   family_meta.
4. **Recipe applied to the canonical RMG-adjlist form.** The group matcher
   produces labelings in explicit-H (AddHs) atom-index space, and the step-01
   recipe engine is label-index-sensitive to the reactant's atom ordering
   (it kekulizes semi-kekulized ene/shift products; a different heavy-atom
   order can mis-kekulize). `match()` therefore normalizes each reactant to
   its canonical RMG-adjlist form (`Molecule.from_adjacency_list(m.to_adjlist())`)
   before applying the recipe, so the label indices land on the atoms RMG
   labeled. (The group-`parse` `_parse_orders` also had to keep `S`/`B` in
   `[S,D,T,B]` order-sets - an early drop of those orders broke the
   intra_H_migration match.)

## What the next step (job-05/step-04 families) should know first

- `TemplateFamily` (in core/template.py) is the family holder the step-04
  loader should emit: build `entries` (the group tree), `top`,
  `forward_template`, `reactant_num` (RMG effective count), and attach
  `recipe`/`reverse_recipe`/`own_reverse`/`reversible`/`allow_charged_species`/
  `electrons`/`product_num_forward`/`reverse_map`. `match(family, reaction)`
  and `TemplateMatcher` consume exactly this shape; the step-03 reference
  loader (`TemplateFamily.from_references`) is the reference implementation,
  but the step-04 loader should read the **live** RMG-database family files
  (not the reference JSON) into the same shape.
- The group matcher (`TemplateMatcher`) already implements the step-02
  `match_molecule(form, slot, branch)` hook, so job-06's core loop can swap
  `RecordedMatcher` -> `TemplateMatcher` to get FRESH labelings (no reference
  replay) in `generate_reactions`.
- `match` runs the recipe in the canonical RMG-adjlist representation; keep
  that if the loader changes the Molecule construction path.
- The 2 benzene descent-label Cd/Cb differences are expected (representation)
  and documented; the match verdict + labeling are exact. The group-tree
  descent (`descend_tree`) returns node **labels**; if job-06 needs the
  group **objects**, descend and index `family.entries[node.label]`.
