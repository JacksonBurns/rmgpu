# job-05/step-05 report - Job-05 gate (product enumeration parity)

Date: 2026-09-01 (EDT)
Status: **RED (valid gate outcome - 31/32 exact, 1 root-caused documented
mismatch; reverse + timing green)**

## What was built

- `gates/gate_05.py` - the job-05 gate. Runs rmgpu's real
  `generate_reactions` (fresh path: step-03 group matcher, step-01 recipe
  engine, step-02 degeneracy collapse) on the 32-case set and compares
  against the recorded RMG-Py ground truth: product SETS (canonical
  explicit-H kekulized SMILES) + per-product degeneracy, EXACT; reverse
  recovery for own-reverse reversible families (re-enumerate a fwd
  reaction's products through the same family, check the original
  reactants are recovered); timing per case (floor 5 s on the max, the
  case set's largest reactants are 25 atoms explicit, stronger than the
  brief's 10-atom requirement). Records the blocked-families list from
  the loader. Exit 0 = GREEN, 1 = RED. Writes
  `reports/gate_05_results.json`.
- `reports/gate_05_results.json` - machine-readable gate output (per-case
  parity, got/want product maps, reverse per-reaction, timing, blocked).
- (carried in from the prior session of this step, committed here)
  `gates/gate05_cases.py` - the fixed 32-case (family, reactants) set
  (every 'default'-set family in the c3h4/superminimal mechanisms + the
  step-02/03 test fixtures);
  `scripts/record_job05_step05_reference.py` - the RMG-Py ground-truth
  recorder (runs rmgpy's own `KineticsFamily.generate_reactions` in
  rmg_env; non-circular - no rmgpu code);
  `gates/baselines/job05/step05_gate_reference.json` - the recorded
  reference (32 cases, fwd+rev collapsed reactions, raw counts,
  family_meta, reverse checks);
  `rmgpu/core/enumeration.py` - the fresh-path wiring this gate drives:
  `forbidden` on the enumeration Family, RMG
  `ForbiddenStructures.is_molecule_forbidden` port (label-aligned subgraph
  check, run on labeled reactants BEFORE the recipe and on each product
  piece AFTER - the check that drops the diradical-forming H_Abstraction
  applications), `_clean_explicit` helper, per-reaction template labels
  in `_enumerate_fresh` (RMG `get_reaction_template_labels` - isomorphic
  products with different templates stay separate in
  `find_degenerate_reactions`);
  `rmgpu/molecule/resonance.py` - two root-cause fixes (job-01 module,
  surfaced by the gate): `_get_lone_pairs` now uses RMG-Py's exact
  formula on the explicit-H bond order (the per-element heuristic wrongly
  gave carbon lone pairs, firing the lone-pair-radical generator on
  hydrocarbon radicals - a product-set divergence);
  `_generate_allyl_delocalization_resonance_structures` now delocalizes a
  radical exocyclic to an aromatic ring (benzylic) into the ring via
  aromatic ring bonds and the aromatic ipso carbon's single ring bond,
  with aryl radicals guarded out and invalid shifts (pentavalent target)
  dropped at kekulization.
- ~80 `scripts/_chk_*` / `_dbg_*` / `_probe_*` scratch scripts (root-cause
  investigation; kept per repo precedent, job-05/step-04 committed the
  same kind).

## Checks run (real results)

1. `/home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_05.py`
   (exit 1 = RED, 80 s):
   ```
   == JOB-05 GATE: product-enumeration parity vs RMG-Py ==
   cases: 32 (reference 32)
   parity exact (set + degeneracy): 31/32
   reverse recovery (in scope): 37/37 ok
   timing max: 0.3772 s (floor 5.0 s): OK
   blocked families: 0 -
   MISMATCH Intra_ene_reaction C[CH]C1=CC=CC=C1  n_rxn=4 ref=2
   GATE STATUS: RED
   ```
   Full per-case listing and the got/want product maps in
   `reports/gate_05_results.json`.

2. `pytest tests/ -q` (the rmgpu env): **595 passed**, 68 warnings, ~20 s.
   (Includes the job-01 resonance tests; the resonance.py fixes above
   change no test expectation - the resonance test is a superset check.)

## The one mismatch (root-caused, documented)

`Intra_ene_reaction` on `C[CH]C1=CC=CC=C1` (the 1-phenylethyl radical):

- rmgpu got: 3 spurious products (each deg 1.0:
  `C=CC1=C=CC=C[CH]1`-type allenes, formula C9H10) +
  `C=CC1=CCC=C[CH]1` at **3.0**.
- RMG wants: `C=CC1=CCC=C[CH]1` (A) at **6.0** and
  `C=CC1=CC[CH]C=C1` (B) at **3.0**.

Root cause (all in the **job-01 resonance/matcher modules**, verified
against RMG-Py in rmg_env), common cause = aromatic resonance handling of
a benzylic radical:

1. **Resonance form set.** RMG-Py generates **5** forms for the 1-phenylethyl
   radical: the aromatic form, **two** ortho ring-radical forms
   (`CC=C1[CH]C=CC=C1`, the two symmetric ortho sites), the **para**
   ring-radical form (`CC=C1C=C[CH]C=C1`), and the input kekulized
   benzylic form (`C[CH]C1=CC=CC=C1`). rmgpu generates **3**: the aromatic
   form, ONE ortho form, and the kekulized benzylic form - it is missing
   the second (symmetry-equivalent) ortho form and the **para** form.
   Product B comes ONLY from the para form, so rmgpu cannot produce it;
   and A's degeneracy is 3.0 instead of 6.0 because RMG counts the two
   ortho applications.
2. **The kekulized benzylic form must not be reacted.** RMG-Py's
   `filter_resonance_structures` (rmgpy/molecule/filtration.py) drops
   non-aromatic kekulized structures whose 6-ring is a standard
   SDSDSD/DSDSDS ring (redundant with the aromatic form) and
   `mark_unreactive_structures` then sets `reactive=False` on the
   filtered-out original (verified: RMG form 4 has `reactive=False`).
   RMG's `_generate_reactions` unimolecular branch skips non-reactive
   forms (`if molecule.reactive or react_non_reactive`). rmgpu has no
   `reactive` flag and reacts the kekulized form - its 3 allene products
   are exactly this form's output (verified per-form in this session:
   forms aromatic -> 0 RMG matchings, ortho -> 3, para -> 6, kekulized ->
   3 allene raws).
3. **Aromatic bond order in the matcher.** RMG keeps aromatic ring bonds
   as 1.5 in the group matcher, so the Intra_ene template's `[D,T]`
   (2.0/3.0) bonds do NOT match an aromatic ring (RMG aromatic form -> 0
   matchings, verified). A matcher that kekulizes aromatic bonds to
   1.0/2.0 before comparing spuriously matches them. (Attempted a
   1.5-reporting fix in `rmgpu/molecule/group.py` this session; it broke
   a step-03 parity test and did not close this case - reverted. The
   clean fix belongs with the resonance form-set fix above.)

All three fixes are in job-01's modules (`resonance.py`, the matcher's
aromatic handling, and a `reactive` flag in the enumeration loop) - the
step file scopes this step's in-place fixes to recipe.py/template.py, so
this mismatch is documented here rather than fixed in this step.
**Recommendation for the next session: a follow-up fix step (job-01-style,
cf. job-01/step-08) covering (1) the para + second-ortho resonance forms
for benzylic/aromatic radicals, (2) RMG's `reactive` flag (filter + skip),
(3) the matcher's aromatic 1.5 bond handling - then re-run gate_05.py
(expected: 32/32, GREEN).**

## Reference reads beyond the list

The step file lists no new reference reads. To root-cause the one mismatch
(allowed: "fix a common-cause mismatch ... root-cause it"), read in
rmg_env/RMG-Py (read-only): `rmgpy/data/kinetics/family.py`
(`_generate_reactions` unimolecular branch, `_create_reaction`,
`get_reaction_template_labels`), `rmgpy/molecule/filtration.py`
(`filter_resonance_structures`, `mark_unreactive_structures`,
`check_reactive`), `rmgpy/molecule/molecule.py`
(`find_subgraph_isomorphisms`), `rmgpy/molecule/resonance.py` (the
`reactive` guard). No reads were needed for the passing 31 cases.

## Deviations

- The recorded reference's `reverse_checks` section is empty of usable
  data: all 35 entries carry a recorder-side
  `AttributeError: 'list' object has no attribute 'is_isomorphic'`
  (the recorder passed a list where a Species was expected in its reverse
  section). The gate therefore implements the reverse check itself,
  in-rmgpu (37 in-scope fwd reactions, all recover their reactants). The
  reference's fwd product/degeneracy ground truth (what the gate must
  match) was unaffected and is used as-is.
- `gates/gate05_cases.py` + the reference JSON + the recorder were
  produced in the prior session of this same step (before the context
  break) and are committed with this step; they match the step file's
  deliverable spec (case set = default-set families in the
  c3h4/superminimal mechanisms + the RMG-Py fixtures).

## What the next step should know first

1. **The gate is re-runnable in ~80 s** (`python gates/gate_05.py`); after
   the resonance fix it should go GREEN (32/32) without touching the gate.
2. **job-05's machinery is otherwise exact**: 31/32 cases with exact
   product sets AND exact per-product degeneracies, 37/37 reverse
   recovery, 0 blocked families, max enumeration 0.38 s. The fresh path
   (group matcher + recipe + forbidden check + per-reaction templates +
   degeneracy collapse) is the one job-06 will drive.
3. The Intra_ene fix plan (3 items) is in the mismatch section above;
   item 2 (the `reactive` flag) is a small change in
   `rmgpu/core/enumeration.py` (expand_resonance / the fresh enumeration
   loop can carry a per-form flag mirroring RMG's
   `ensure_independent_atom_ids`), items 1 and 3 are in
   `rmgpu/molecule/resonance.py` / `rmgpu/molecule/group.py`.
4. The recorded reference's `reverse_checks` are unusable (recorder bug,
   see Deviations) - if a future step wants RMG-side reverse ground
   truth, fix `scripts/record_job05_step05_reference.py` (pass Molecules,
   not lists, to `check_for_same_reactants` in the reverse section) and
   re-record.
