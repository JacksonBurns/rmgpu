# job-05/step-06: Fix the Intra_ene resonance-form gap (job-05 gate RED)

Job: job-05 - Reaction recipe DSL + product enumeration
Prereq: job-05/step-05 gate is RED (31/32; one root-caused mismatch)
This step fixes the single RED case from the gate before job-05 can close.

## Goal

Fix the one product-set/degeneracy mismatch found by gates/gate_05.py:
`Intra_ene_reaction` on `C[CH]C1=CC=CC=C1` (the 1-phenylethyl radical).
rmgpu currently emits 3 spurious allene products (deg 1.0 each) + A at
3.0; RMG-Py emits A (`C=CC1=CCC=C[CH]1`) at 6.0 and B
(`C=CC1=CC[CH]C=C1`) at 3.0. Root cause (verified in rmg_env; full
analysis in reports/job-05-step-05-gate.md, "The one mismatch"): the
job-01 resonance/matcher handling of benzylic radicals. Three fixes, all
faithful to RMG-Py:

1. **Resonance form set** (rmgpu/molecule/resonance.py): the benzylic
   radical must delocalize to BOTH ortho ring positions AND the para
   position. RMG-Py's set: aromatic, ortho x2, para, kekulized-benzylic
   (5 forms). rmgpu today: aromatic, one ortho, kekulized-benzylic (3).
   The current `_generate_allyl_delocalization_resonance_structures`
   already covers the aromatic-form -> one-ortho path; extend the
   candidate pi-bond collection so a radical bonded to an aromatic ipso
   carbon considers BOTH ring bonds of the ipso atom (it sees one DOUBLE
   and one SINGLE in the kekulized form) and, from each ortho form, the
   para form (the radical at an ortho carbon delocalizes through the
   ortho-para bond; the ipso carbon's exocyclic C=C keeps it valid).
   Invalid shifts (pentavalent target, e.g. radical onto the ipso
   carbon) must be dropped at kekulization (already implemented).

2. **The `reactive` flag** (rmgpu/core/enumeration.py, mirroring
   rmgpy/molecule/filtration.py + RMG _generate_reactions): resonance
   forms that are non-aromatic kekulized variants of an aromatic
   structure (6-ring with SDSDSD/DSDSDS bond orders, and the aromatic
   form present in the same species) are filtered out, and the
   filtered-out original form is kept in the list but marked
   `reactive=False` and SKIPPED by the enumeration loop (RMG
   `if molecule.reactive or react_non_reactive`). Implement the filter
   + flag where rmgpu expands resonance (expand_resonance /
   assign_fresh_ids or a new helper) and check the flag in
   `_enumerate_fresh`'s per-form loop. This is what stops the kekulized
   benzylic form from generating the 3 allene products.

3. **Aromatic 1.5 bond orders in the group matcher**
   (rmgpu/molecule/group.py `_explicit_graph` or the bond comparison):
   RMG-Py compares aromatic ring bonds as 1.5, so `[D,T]`/`[S,D]`
   template bonds (2.0/3.0) do NOT match an aromatic ring (the aromatic
   1-phenylethyl form must match the Intra_ene template 0 times; in
   rmg_env: RMG form 0 -> 0 matchings, forms 1/2/3 -> 4/6/5). rmgpu's
   molecules are kekulized on construction, so the explicit graph must
   restore 1.5 for aromatic ring bonds before comparison. NOTE: a naive
   "report 1.5 for aromatic bonds" change broke a step-03 parity test
   (the kekulized benzene cases) - the fix must distinguish the aromatic
   (delocalized) representation from a genuinely kekulized structure, or
   the change must be scoped so kekulized-form matching is unaffected.
   Verify against the full step-03 reference (test_template_match.py
   322/322) after the change.

The three fixes are interdependent: (1)+(2) alone give A at 3.0 still
missing the ortho-duplicate and B (needs para); (2) alone removes the
spurious allenes but keeps A at 3.0 and drops B. All three are needed
for 32/32.

## Reference to read

- RMG-Py rmgpy/molecule/filtration.py: filter_resonance_structures,
  mark_unreactive_structures, check_reactive (the exact filter rules).
- RMG-Py rmgpy/molecule/resonance.py: how the aromatic-radical forms are
  generated (ortho x2 + para for benzylic; aryl radicals not shifted).
- RMG-Py rmgpy/molecule/molecule.py + group.py: how aromatic bond orders
  (1.5) enter is_subgraph_isomorphic / bond comparison.
- reports/job-05-step-05-gate.md: the per-form matching counts recorded
  in this job (RMG: forms 1/2/3 -> 3/6/5 raw matchings, form 0 -> 0;
  rmgpu today: kekulized form -> 3 allene raws).

## Deliverables

- The three fixes (resonance.py, enumeration.py, group.py as scoped
  above) + any test updates (resonance test is a superset check;
  step-03 template tests must stay 322/322).
- reports/job-05-step-06-fix-intraene.md.

## Checks (must run and pass before claiming done)

```
/home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest tests/ -q   # all pass
/home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_05.py      # GREEN (32/32)
```

## Pitfalls

- Do NOT change the canonicalization or the degeneracy-collapse logic to
  make the case pass - the fix is the resonance form set + reactive
  flag + matcher aromatic handling (the responsible modules).
- The para shift must validate at kekulization: the radical landing on
  the ipso carbon is pentavalent and must be dropped (RMG-Py does not
  generate that form either).
- Fix 3 is the riskiest: kekulized aromatic forms (e.g. benzene
  `C1=CC=CC=C1` in the R_Addition_MultipleBond case) must still match
  their templates exactly as today (31/32 must stay 31/32 until fix 1+2
  land, then 32/32). Re-run gate_05.py after EACH of the three fixes.
- ONE session at a time; no subagents; env python only.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-05/step-06: <summary>"` (NO push).
2. STATUS.md: set the 05/06 step row to done, update the job-05 gate row
   (05/05) to done (gate GREEN), append a session-log entry
   (`### <date> - job-05/step-06 / built: ... / checks: ... (GREEN|RED +
   one line) / commits: <hashes> / next: <next step id>`), and update the
   top-level NEXT pointer to the next step's file.
3. Write reports/job-05-step-06-fix-intraene.md (files changed + 1 line
   each, the checks + real results, reference reads, deviations, what the
   next step should know first).
4. STOP. Do not start the next step. Do not spawn subagents.
