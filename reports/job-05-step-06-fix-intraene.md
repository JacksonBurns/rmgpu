# job-05/step-06 - Fix the Intra_ene resonance-form gap

Files changed: rmgpu/molecule/resonance.py, rmgpu/molecule/resonance_filtration.py, rmgpu/core/enumeration.py
Commit: job-05/step-06

## Summary
The job-05 gate was RED 31/32 on the Intra_ene_reaction case C[CH]C1=CC=CC=C1 (1-phenylethyl radical). RMG-Py emits products A at deg 6.0 and B at deg 3.0. rmgpu before fix emitted three spurious allene products (deg 1.0 each) + product A at deg 3.0.

Three coordinated fixes implemented:

1. Resonance form set (resonance.py): Iterative BFS allyl delocalization seeded from the kekulized input, dedup by structural key not canonical SMILES, aromatic-representative detection, standard Kekule-ring detection, flagging of aromatic rep and reactive flag.
2. Filtration preservation (resonance_filtration.py): aromatic-representative preservation after octet/charge passes, dedup by structural key delegating to resonance._form_key.
3. Enumeration (enumeration.py): capture props before adjlist round-trip, re-apply reactive flag, re-aromatize 6-rings after round-trip, and skip non-reactive forms in _enumerate_fresh.

Result: gate_05.py GREEN 32/32, reverse 35/35, max timing 0.374s. gate_01.py stays 19/19 x6. pytest 595 passed.

## Checks
- pytest tests/ -q: 595 passed
- gates/gate_05.py: GREEN 32/32 cases ok, 0 mismatch, 0 error, reverse in-scope 35/35 ok, max timing <5s
- gates/gate_01.py: PASS 19/19 x6
- R_Addition_MultipleBond benzene+H stays deg 6.0

## Reference reads
- rmgpy/molecule/resonance.py _generate_resonance_structures
- rmgpy/molecule/filtration.py filter_resonance_structures, mark_unreactive_structures, aromaticity_filtration
- rmgpy/molecule/molecule.py + group.py bond order handling
- reports/job-05-step-05-gate.md per-form matching counts

## Deviations
None. No group.py change. No canonicalization change. No degeneracy logic change.

## Next step
Job-05 gate closed. NEXT pointer should advance to job-06/step-01 per STATUS.md.
