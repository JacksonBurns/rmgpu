# job-01/step-08: Fix parity failures (job-01 gate)

## Result: GATE GREEN

Both required checks pass:

- `pytest tests/ -q` -> **118 passed**
- `gates/gate_01.py` -> **GATE STATUS: PASS**, all 6 checks 19/19:
  - adjlist_roundtrip 19/19
  - adjlist_parity 19/19
  - smiles_parity 19/19
  - atomtype_parity 19/19
  - resonance_parity 19/19
  - symmetry_parity 19/19

## What was done

This step picked up partial, uncommitted work from two prior step-08 sessions
(the first made progress, the second made none due to a misdiagnosed interpreter
path - see STATUS.md session log). All prior partial fixes were uncommitted;
this session rebuilt, verified against RMG-Py, and completed the parity fixes.

### Root causes found and fixed

1. **Test set label mismatch (prior-session artifact).** `gates/test_set.py`
   had a label list shifted by one relative to `TEST_MOLECULES`
   (`methoxy_radical` did not exist). The committed baselines had been generated
   from the shifted labels, so every molecule was being compared against the
   wrong reference. Fixed the labels and **regenerated all baselines** with
   `gates/generate_references.py` in the `rmg_env` (real RMG-Py) environment.
   The gate is now comparing against true RMG-Py output.

2. **adjlist round-trip + parity (0/19 -> 19/19).** Rewrote
   `rmgpu/molecule/adjlist.py` as a faithful port of RMG-Py
   `adjlist.py`:
   - serialization and parsing operate on the **explicit-hydrogen** graph
     (`Molecule._with_explicit_h()`), matching RMG-Py's vertex set;
   - `u<N> p<N> c<C>` emitted and parsed as separate space-delimited tokens;
   - lone pairs computed with RMG-Py's valence formula
     `(valence - unpaired - charge - bond_order) // 2` (H/Li forced to 0);
   - column widths and ordering mirror `to_adjacency_list()`.
   `Molecule.to_adjlist()`/`from_adjacency_list()` updated accordingly
   (the parser no longer double-adds H). Unknown element symbols now raise
   `InvalidAdjacencyListError` (validated in `from_adjacency_list`) instead of
   leaking a raw RDKit `RuntimeError`.

3. **atomtype parity (1/19 -> 19/19).** Rewrote `rmgpu/molecule/atomtype.py`
   as a faithful port of RMG-Py `atomtype.py`:
   - assignment runs on the explicit-H graph (RMG-Py assigns types to every
     vertex including hydrogens, e.g. ethane -> `Cs Cs H0*6`);
   - feature extraction (bonds by order, lone pairs, charge, radical electrons)
     and first-match type tables mirror the RMG-Py database, including the
     previously missing Si/P/S/O4b and wildcard `[]` match rules.
   Verified line-for-line against RMG-Py on all 30 step-05 test molecules and
   all 19 gate molecules.

4. **symmetry parity (4/19 -> 19/19).** Rewrote `rmgpu/molecule/symmetry.py`
   as a faithful port of RMG-Py `symmetry.py`:
   - all factors computed on the explicit-H graph;
   - atom, bond, axis (cumulated double-bond), and cyclic factors ported
     verbatim (bond factors applied once per bond, `i < j`; no dedup of
     equivalent axes);
   - RDKit API differences handled (no `IsAtomInAnyRing`/`GetDegreeOfAtom`/
     `Atom.GetBondBetweenAtoms` in this build).
   All 19 gate values and all step-02c test values now match RMG-Py exactly
   (methane 12, ethane 18, water 2, ethylene 4, benzene 12, isobutane 81, ...).

5. **smiles parity (17/19 -> 19/19).** NO2 and HNO3-like (nitrite) disagree
   because RMG-Py canonicalizes any N/S-containing species with **OpenBabel**
   (`[O-][N+]=O`) while rmgpu used RDKit (`O=[N+][O-]`).
   - Installed `openbabel` (3.2.1) into the `rmgpu` conda env from
     conda-forge (only other package touched: openssl, a dep).
   - Ported RMG-Py `translator.to_smiles` behavior into `Molecule.to_smiles()`:
     formula-based `MOLECULE_LOOKUPS`/`RADICAL_LOOKUPS` first, then OpenBabel
     canonicalization for N/S species, RDKit fallback otherwise. OpenBabel is
     optional at runtime (graceful fallback if absent).

6. **resonance parity (17/19 -> 19/19).** Two fixes:
   - **Charge bookkeeping bug** in
     `_generate_adj_lone_pair_radical_resonance_structures`: the radical site
     gains a lone pair (charge **-1**) and the lone-pair site gains the radical
     (charge **+1**). rmgpu had this backwards, so NO2's `[O]N=O` resonance
     form was never generated.
   - **Missing filtration.** Ported RMG-Py `filtration.py` (octet-deviation ->
     charge-span -> electronegativity/proximity stabilization) to a new module
     `rmgpu/molecule/resonance_filtration.py` and applied it in
     `generate_resonance_structures()`, which drops `[O-]N=[O+]` (higher charge
     span, no new radical/multiple-bond site) leaving exactly
     `['[O-][N+]=O', '[O]N=O']` as in RMG-Py. The input structure is always
     preserved. (Note: the pre-existing `rmgpu/molecule/filtration.py` is an
     unrelated step-06 forbidden-structure module and was left untouched -
     the resonance port lives in the separate `resonance_filtration.py`.)

### Tests updated (were stale, encoding the old buggy behavior)

- `tests/test_atomtype.py` - atom types are H-inclusive with RMG-Py labels
  (`Cs`, `O2s`, `N3s`, ...); length checks compare against the explicit-H graph.
- `tests/test_symmetry.py` - ethane 3 -> 18, ethylene 2 -> 4 (RMG-Py values).
- `tests/test_adjlist.py` - now passes without changes (round-trip works).
- `tests/test_resonance.py` (step-05) - pre-existing flaw: the "aromatic form"
  selector matched the first baseline line containing *any* lowercase letter,
  picking the hybrid Kekule form `C1=Cc2ccccc2C=C1` instead of the fully
  delocalized `c1ccc2ccccc2c1` (which rmgpu does generate). Fixed the selector
  to require all-lowercase atom symbols, matching the test's own stated intent.

## Files changed

- `rmgpu/molecule/adjlist.py` (new - full RMG-Py port)
- `rmgpu/molecule/atomtype.py` (new - full RMG-Py port)
- `rmgpu/molecule/symmetry.py` (rewritten - full RMG-Py port)
- `rmgpu/molecule/molecule.py` (explicit-H helpers, to_adjlist/from_adjacency_list,
  to_smiles with OpenBabel + formula lookups, get_formula sanitize, element validation)
- `rmgpu/molecule/resonance.py` (adj lone-pair-radical charge fix + filtration)
- `rmgpu/molecule/resonance_filtration.py` (new - RMG-Py filtration port)
- `gates/test_set.py` (label list corrected)
- `gates/baselines/job01/results.json` (regenerated from real RMG-Py)
- `tests/test_atomtype.py`, `tests/test_symmetry.py`, `tests/test_resonance.py`
- `reports/job-01-step-08-fix-parity.md` (this file)
- `STATUS.md`

## Environment change

- `rmgpu` conda env: added `openbabel 3.2.1` (conda-forge) + its deps
  (libxml2, xorg-lib*, pcre2, openssl update). Required for RMG-parity
  canonical SMILES of N/S species. Recorded here so the env is reproducible.

## Notes for the human

- The job-01 gate (the milestone gate) is now GREEN. Job-01 may close.
- Untracked `debug_*.py` probes left by earlier sessions remain in the repo
  root (deletion requires your consent). Recommend deleting them or adding to
  .gitignore.
- `gates/dump_resonance.py` and `scripts/atomtype_reference.py` (from steps
  03/05) are untracked helpers; `gates/baselines/` (needed by gate_01 and
  test_resonance) should be committed.
