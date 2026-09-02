# job-05/step-06: Fix the Intra_ene resonance-form gap (job-05 gate RED)

Job: job-05 - Reaction recipe DSL + product enumeration
Prereq: job-05/step-05 gate is RED (31/32; one root-caused mismatch)
This step fixes the single RED case from the gate before job-05 can close.

This file is written to be executed IN ONE SHOT by a fresh agent that has no
other context. It encodes the full mechanism, the RMG-Py ground truth, every
RDKit/RDG pitfall discovered while solving it, and the exact checks with their
expected values. A previous agent got partway and left a partially-correct,
UNCOMMITTED implementation in the working tree (`git diff`); read it, but do
NOT trust it blindly - it contains two real bugs that are called out below.
The verified, working design is fully specified here so you can reproduce or
validate it end to end.

## The RED case (what must turn GREEN)

`gates/gate_05.py`, family `Intra_ene_reaction`, reactant `C[CH]C1=CC=CC=C1`
(the 1-phenylethyl radical, the benzylic radical).

- rmgpu (before fix): 3 spurious allene products (deg 1.0 each) + product A at
  deg 3.0.
- RMG-Py (target, verified in rmg_env):
    - A = `C=CC1=CCC=C[CH]1` at deg **6.0**
    - B = `C=CC1=CC[CH]C=C1`  at deg **3.0**
  (No allenes.)

Gate must end at **32/32 cases ok, 0 mismatch, 0 error, reverse in-scope
35/35 ok, max timing < 5s** (observed ~0.38s).

## Root cause (verified in rmg_env + rmgpu)

The benzylic radical must delocalize into the ring. RMG-Py generates a
**5-form resonance set** for 1-phenylethyl; rmgpu generated only 3 (missing the
2nd ortho and the para), and rmgpu's matcher treated the kekulized form's ring
as S/D so it matched the `[S,D]`/`[D,T]` Intra_ene template and emitted the
allenes. Three coordinated fixes, all RMG-faithful, are required. They are
interdependent: (1)+(2) alone leave A at 3.0 and no B; (2) alone removes the
allenes but keeps A at 3.0 and drops B. **All three are needed for 32/32.**

## RMG-Py ground truth (verified against /home/jackson/rmg/RMG-Py in rmg_env)

Use these as the acceptance oracle. `Molecule.generate_resonance_structures(keep_isomorphic=True)`:

- **1-phenylethyl** `C[CH]C1=CC=CC=C1` -> 5 forms (canonical SMILES shown; flags in parens):
    1. `C[CH]c1ccccc1`      (reactive=True,  aromatic-representative)
    2. `CC=C1[CH]C=CC=C1`   (reactive=True,  ortho)
    3. `CC=C1C=C[CH]C=C1`   (reactive=True,  para)
    4. `CC=C1[CH]C=CC=C1`   (reactive=True,  ortho - isomorphic to form 2, different radical position, KEPT)
    5. `C[CH]C1=CC=CC=C1`   (reactive=False, the kekulized benzylic - the allene-maker, SKIPPED)
  Per-form Intra_ene `[S,D]`/`[D,T]` matchings (RMG): form1 (aromatic) -> **0**,
  forms 2/3/4 (ortho/para) -> 4/6/5, form5 (kekulized, non-reactive) -> not reacted.
- **benzene** `c1ccccc1` -> `['c1ccccc1', 'C1=CC=CC=C1']` (aromatic + kekulized, 2 forms).
- **toluene** `Cc1ccccc1` -> `['Cc1ccccc1', 'CC1=CC=CC=C1']` (2 forms).
  These two are what the job-01 gate reference (`gates/baselines/job01/results.json`
  `resonance_parity`) expects. **Do not regress them.**
- **R_Addition_MultipleBond** benzene+H stays deg 6.0 after the fix (the
  kekulized benzene form matches the `[D,T,B]` template; the aromatic form also
  matches via `[D,T,B]`; both dedupe to one reaction).

## The three fixes (exact mechanism)

### Fix 1 - Resonance form set: `rmgpu/molecule/resonance.py`

The benzylic radical must reach BOTH ortho positions AND para. The existing
single-pass `_generate_allyl_delocalization_resonance_structures` reaches only
ONE ortho from the kekulized input (the ipso double-bond shift). The fix: run
the allyl shift **ITERATIVELY (BFS)** and **seed from the kekulized input**.

Why the seed must be kekulized (NOT the aromatic representative): a shift that
converts an aromatic ring bond to a single bond leaves RDKit's `Kekulize` unable
to resolve the ring (the other ring bonds are still aromatic-flagged), so every
candidate is dropped and a BFS from an aromatic seed yields ZERO forms. From the
kekulized input, the ipso double-bond shift is valid (one ortho), and from each
ortho form the para shift is valid. This mirrors RMG-Py, which applies each
resonance method to every generated form.

The allyl shift itself (already present) must:
- consider BOTH ring bonds of an aromatic ipso carbon (one DOUBLE, one SINGLE
  in the kekulized form) as delocalization paths (reaches both ortho);
- keep the radical atom guard NON-aromatic (an in-ring aryl radical is already
  delocalized and must not break its ring);
- build the candidate (radical to far atom, pi bond -> single, new double bond
  radical-site/near-end, clear aromatic flags on touched bonds), then validate
  with `Chem.SanitizeMol` + `Chem.Kekulize(..., clearAromaticFlags=True)`.
  NOTE: this rdkit build's `Kekulize` returns `None` (not True) on success, so a
  **raised exception** is the only failure signal - wrap in try/except and
  `append` in the `else:`. An invalid shift (radical onto the ipso carbon,
  pentavalent) raises and is dropped (RMG-Py does not generate that form either).

**Dedup by a STRUCTURAL key, NOT canonical SMILES.** The two ortho forms share a
canonical SMILES (different radical POSITION) and the aromatic-representative +
kekulized form share a canonical SMILES too. SMILES dedup collapses all of
these and gives the wrong set. The structural key (below) keeps them distinct.

### Fix 1b - The structural key + aromatic-representative detection: `rmgpu/molecule/resonance.py`

These are the subtle, bug-prone helpers. Implement exactly as described.

`_form_key(m)` - the dedup key:
```
em = m._rdkit                      # the STORED mol (implicit-H). CRITICAL - see pitfall P1.
bonds = []
for b in em.GetBonds():
    o = b.GetBondTypeAsDouble()
    if b.GetIsAromatic():
        o = 1.5                   # normalize: aromatic-flagged bond == 1.5
    if o in (1.0, 1.5, 2.0, 3.0):
        bonds.append((min(a1,a2), max(a1,a2), o))
atoms = [(a.GetIdx(), a.GetSymbol(), a.GetNumRadicalElectrons(), a.GetFormalCharge())
         for a in em.GetAtoms()]
return (tuple(sorted(bonds)), tuple(atoms))
```
- The aromatic-flag normalization is what makes the two equivalent aromatic
  representations (the 1.5 `SetAromaticity` form and the flagged S/D `c1ccccc1`
  parse) collapse to ONE key (they are the same representative), while the
  genuinely-kekulized variant (S/D, NO aromatic flag) stays distinct.
- MUST be computed on the STORED mol `m._rdkit`, **not** `m._with_explicit_h()`
  (P1).

`_aromatic_rep(mol)` -> `(is_aromatic_rep, ring_bond_pairs)` - computed on
`mol._rdkit` (stored mol, P1):
- `is_aromatic_rep` is True iff there is a 6-membered ring whose SIX bonds all
  carry `bond.GetIsAromatic()` (the delocalized aromatic representative). RDKit
  shows this as either 1.5 bonds OR S/D bonds WITH the aromatic flag - the test
  is the FLAG, not the bond order. The genuinely-kekulized variant (S/D, no flag)
  is NOT an aromatic representative.
- `ring_bond_pairs` = frozenset of `(min(a,b), max(a,b))` ring bonds of that ring
  (empty if none). (Captured here for completeness; the Fix-3 re-aromatization
  actually re-detects the ring itself, so this is optional.)

`_has_standard_kekule_ring(mol)` - computed on `mol._rdkit` (stored mol, P1):
- True iff a 6-ring has alternating S/D bond orders (`SDSDSD` or `DSDSDS`) AND
  NOT all six bonds are aromatic-flagged (i.e. it is a genuinely kekulized
  variant, not the aromatic representative). This is RMG filtration's
  "redundant Kekule variant" criterion - the form that gets marked non-reactive.

`_set_resonance_flags(structures)` - attach the two RMG-Py resonance flags to
each form's RDKit mol (the enumeration loop and matcher read these):
```
has_aromatic = any(_form_is_aromatic(f) for f in structures)
for f in structures:
    rdmol = f._rdkit
    if _form_is_aromatic(f):
        rdmol.SetProp('rmgpu_aromatic_rep', '1')
    elif rdmol.HasProp('rmgpu_aromatic_rep'):
        rdmol.DelProp('rmgpu_aromatic_rep')
    if not rdmol.HasProp('rmgpu_reactive'):
        rdmol.SetProp('rmgpu_reactive', '1')
    if has_aromatic and not _form_is_aromatic(f) and _has_standard_kekule_ring(f):
        rdmol.SetProp('rmgpu_reactive', '0')     # the redundant kekulized variant
```
Flags live on the RDKit mol (not the Molecule wrapper) because the enumeration
path rebuilds every form through the adjlist round-trip, which drops plain
Python attributes but the caller re-applies the mol props (Fix 2 below).

`generate_resonance_structures(molecule)` - wire it up:
- if radical: `new_structures.extend(_allyl_bfs([molecule.copy()]))` (BFS, kekulized seed)
- if hasLonePairs and not is_aromatic: the lone-pair generators (unchanged)
- if is_aromatic: extend with `_generate_optimal_aromatic_resonance_structures`
  (1.5 form) and `_generate_kekule_structure` (S/D form)
- dedup `all_structures = [molecule.copy()] + ...` by `_form_key` (input first)
- `_set_resonance_flags(all_structures)`
- `result = filter_structures(all_structures, molecule.copy())`
- `_set_resonance_flags(result)`   # re-assert on the survivors (filtration may reorder/drop)
- return `result`

### Fix 1c - Filtration must keep the aromatic representative + the kekulized variant: `rmgpu/molecule/resonance_filtration.py`

Two changes to `filter_structures` (ported from RMG-Py filtration.py):

1. **Aromatic-representative preservation** (RMG's `aromaticity_filtration`,
   which runs only for aromatic species). The octet pass DROPS the delocalized
   aromatic form: `_bond_order` maps a 1.5 aromatic bond to 1 for valence, so a
   benzene-ring carbon looks like valence 5 (octet deviation 3 per atom). Capture
   the aromatic form(s) from the input set before filtration and re-attach them
   after the octet/charge passes if they were dropped. This keeps the 5-form
   benzylic set AND the benzene/toluene aromatic+kek pair.
2. **Dedup by the structural key, NOT canonical SMILES.** Replace the SMILES
   dedup with a dedup on `_structural_key(mol)`, where `_structural_key`
   **delegates to `resonance._form_key`** (single source of truth; lazy import to
   avoid a cycle): `from rmgpu.molecule.resonance import _form_key; return _form_key(mol)`.
   Keeping first occurrence preserves input-first ordering. Still guarantee the
   original is present (insert at 0 if its key is absent).

### Fix 2 - The `reactive` flag: `rmgpu/core/enumeration.py`

Mirror RMG-Py `filter_resonance_structures` + `_generate_reactions`
(`if molecule.reactive or react_non_reactive`). The redundant kekulized variant
is kept in the list but marked `reactive=False` and **skipped** by the
enumeration loop. This is what stops the kekulized benzylic form from emitting
the 3 allene products.

In `_enumerate_fresh`, add a per-form gate:
```
def _is_reactive(form):
    r = form._rdkit
    if r.HasProp('rmgpu_reactive'):
        return r.GetProp('rmgpu_reactive') == '1'
    return True          # RMG default: no flag => reactive
```
and in EVERY per-form loop (unimolecular; bimolecular A+B and B+A; the
single-group split branches) add `if not _is_reactive(form): continue`.
**BUG in the previous draft (BUG-1):** the flag logic was inverted - it returned
True for NON-reactive forms and the loop did `if not _reactive(form): continue`,
which skipped the reactive forms and PROCESSED the non-reactive kekulized form
(the allene source). The gate was still RED (3 allenes). Get the polarity right:
`_is_reactive` returns True when the form SHOULD be reacted.

### Fix 3 - Aromatic 1.5 bond orders: `rmgpu/core/enumeration.py` (NOT group.py)

**KEY LESSON: the fix is NOT in `rmgpu/molecule/group.py`.** The old step file
said to change `_explicit_graph` / the bond comparison, and a naive "report 1.5
for aromatic bonds" change there broke the step-03 parity tests (the kekulized
benzene cases). The correct, RMG-faithful approach: **restore the aromatic
form's genuine 1.5 ring bonds after the adjlist round-trip, and leave the
matcher untouched.** The matcher's bond comparison is an EXACT match on bond
order, and RDKit's AddHs (used by `_explicit_graph`) preserves a genuine
AROMATIC bond as order 1.5 - so once the form's ring bonds are AROMATIC again,
the matcher reports 1.5 for them and `[S,D]`/`[D,T]` templates do not match
while `[D,T,B]` (benzene-including) templates do. No group.py change.

Mechanism in `assign_fresh_ids(species)`:
- The adjlist round-trip (`to_adjlist`/`from_adjacency_list`) **drops the two
  resonance-form props** AND **Kekulizes the delocalized aromatic
  representative's 6-ring to S/D** (P3). So:
    1. **Before** the round-trip, capture each form's `(is_aromatic_rep,
       reactive)` from its RDKit mol props (atom order is preserved by the
       round-trip, so index j maps the same form).
    2. Do the round-trip (existing explicit-H normalization) + shared atom-ID
       assignment (unchanged).
    3. **After** the round-trip, re-apply each form's props, and for any form
       flagged aromatic-representative, **re-aromatize the 6-ring**: a helper
       `_re_aromatize_ring(rdmol)` that sanitizes a copy, finds every 6-membered
       ring, sets each ring bond to `BondType.AROMATIC` (clearing the flag on
       non-ring bonds), sanitizes, then writes the AROMATIC bond types back onto
       the original mol's bonds and re-sanitizes. This restores the 1.5 bonds
       the matcher must see.

Verified result after this: the aromatic form gives **0** Intra_ene matchings,
the ortho/para forms give their matchings, and the kekulized form is skipped by
Fix 2 - yielding exactly A@6.0 + B@3.0.

## Pitfalls (all actually hit; verify you have not reproduced them)

- **P1 - AddHs/sanitize RE-AROMATIZE a genuinely-kekulized benzene ring.**
  `Chem.AddHs`/`Chem.SanitizeMol` on a S/D-no-flag benzene ring re-sets the
  aromatic flag (RDKit re-detects aromaticity). Any aromaticity test or
  structural key computed on `_with_explicit_h()` therefore CANNOT distinguish
  the kekulized variant (S/D, no flag) from the aromatic representative (S/D,
  flagged) - it collapses them and drops the kekulized form (benzene -> 1 form,
  benzylic -> 4 forms: the observed BUG). Compute `_form_key`, `_aromatic_rep`,
  and `_has_standard_kekule_ring` on the **STORED mol** `m._rdkit` (implicit-H),
  which preserves the flag.
- **P2 - canonical SMILES dedup collapses distinct resonance forms.** The two
  ortho forms and the aromatic+kekulized pair each share a canonical SMILES.
  SMILES dedup gives the wrong count. Use the positional structural key.
- **P3 - the adjlist round-trip drops RDKit mol props and Kekulizes the ring.**
  Capture props before, re-apply + re-aromatize after (Fix 3). If you forget
  the re-aromatization, the aromatic form wrongly matches `[S,D]`/`[D,T]`
  (5 matchings instead of 0) and the allenes reappear.
- **P4 - the previous draft seeded the BFS from the aromatic form (zero shifts)
  and had the `_reactive` flag inverted (BUG-1).** Both are fixed as described;
  if you see `C[CH]c1ccccc1` as the only benzylic form, or 3 allenes persisting
  at the gate, you have one of these bugs.
- **P5 - the octet filtration drops the aromatic representative** (spurious
  deviation). Without the aromaticity-preservation re-attach (Fix 1c), the
  aromatic form is absent and the set is 4, not 5.
- **P6 - `Kekulize` returns None on success** in this rdkit build; validate by
  catching the exception, not by checking the return value.
- **Do NOT** change canonicalization or the degeneracy-collapse logic to make
  the case pass. **Do NOT** special-case the matcher for benzene. The fix is the
  resonance set + reactive flag + re-aromatization, in the responsible modules.
- ONE session at a time; no subagents; use the env python
  `/home/jackson/miniforge3/envs/rmgpu/bin/python` only.

## Reference to read (RMG-Py at /home/jackson/rmg/RMG-Py, rmg_env)

- `rmgpy/molecule/resonance.py`: `_generate_resonance_structures` (iterative
  application, the ortho x2 + para for benzylic; aryl radicals not shifted).
- `rmgpy/molecule/filtration.py`: `filter_resonance_structures`,
  `mark_unreactive_structures`, `aromaticity_filtration` (the exact filter
  rules incl. the aromatic-representative preservation).
- `rmgpy/molecule/molecule.py` + `group.py`: how aromatic bond orders (1.5)
  enter `is_subgraph_isomorphic` / bond comparison (why the matcher sees 1.5).
- `reports/job-05-step-05-gate.md`: the per-form matching counts recorded in
  this job (RMG forms 1/2/3 -> 3/6/5 raw, form 0 -> 0; rmgpu kekulized form ->
  3 allene raws).

## Deliverables

- The three fixes across `rmgpu/molecule/resonance.py`,
  `rmgpu/molecule/resonance_filtration.py`, and `rmgpu/core/enumeration.py`
  (group.py is UNCHANGED).
- `reports/job-05-step-06-fix-intraene.md`.

## Checks (must run and pass before claiming done)

Run with `/home/jackson/miniforge3/envs/rmgpu/bin/python`:
```
python -m pytest tests/ -q                 # all pass (baseline 595; must not drop)
python gates/gate_05.py                    # GREEN: 32/32 cases ok, 0 mismatch, 0 error,
                                           # reverse in-scope 35/35 ok, max timing <5s (~0.38s)
python gates/gate_01.py                    # MUST STAY 19/19 x6 (resonance_parity benzene/toluene
                                           #  ->  ['c1ccccc1','C1=CC=CC=C1'] / ['Cc1ccccc1','CC1=CC=CC=C1'])
```
Sanity checks (no allene; A@6.0 + B@3.0):
```
python -c "
import sys; sys.path.insert(0,'/home/jackson/rmgpu/rmgpu')
from rmgpu.core.enumeration import enumerate_reactions
from rmgpu.core.matcher import TemplateMatcher
# (construct the Intra_ene matcher as gates/gate_05.py does for this case) and
# confirm the two products are C=CC1=CCC=C[CH]1 (deg 6.0) and C=CC1=CC[CH]C=C1 (deg 3.0), no allene.
"
```
Also confirm `R_Addition_MultipleBond` benzene+H stays deg 6.0 (it is one of the
32 cases).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-05/step-06: <summary>"` (NO push).
   (Clean up or exclude the `scripts/_probe*.py` diagnostic files before
   committing - they are scratch, not deliverables.)
2. STATUS.md: set the 05/06 step row to done, update the job-05 gate row
   (05/05) to done (gate GREEN), append a session-log entry
   (`### <date> - job-05/step-06 / built: ... / checks: ... (GREEN|RED +
   one line) / commits: <hashes> / next: <next step id>`), and update the
   top-level NEXT pointer to the next step's file.
3. Write reports/job-05-step-06-fix-intraene.md (files changed + 1 line
   each, the checks + real results, reference reads, deviations, what the
   next step should know first).
4. STOP. Do not start the next step. Do not spawn subagents.
