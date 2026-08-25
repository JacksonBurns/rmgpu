# Job-01 Step-07: Job-01 Gate Report

## Summary

Implemented and ran the job-01 gate, comparing rmgpu outputs against RMG-Py
references across 19 test molecules.

## Deliverables

| File | Description |
|------|-------------|
| `gates/gate_01.py` | Gate script comparing rmgpu vs RMG-Py references |
| `gates/test_set.py` | Test molecule set (19 molecules from superminimal, c3h4, minimal) |
| `gates/generate_references.py` | RMG-Py reference generator |
| `gates/baselines/job01/results.json` | RMG-Py reference data |

## Checks Run

### 1. pytest tests/ -q
```
118 passed in 4.52s
```
All existing tests pass.

### 2. python gates/gate_01.py
```
adjlist_roundtrip: 19/19 pass
adjlist_parity: 0/19 pass (19 fail)
smiles_parity: 17/19 pass (2 fail: toluene, NO2)
atomtype_parity: 0/19 pass (19 fail)
resonance_parity: 17/19 pass (2 fail: toluene, NO2)
symmetry_parity: 6/19 pass (13 fail)
GATE STATUS: FAIL
```

## Findings

1. **adjlist_roundtrip (19/19 pass)**: rmgpu adjlist parse/serialize round-trip works correctly.

2. **adjlist_parity (0/19 pass)**: rmgpu adjlist strings differ from RMG-Py. This is a known
   formatting difference and would require byte-level parity work to fix.

3. **smiles_parity (17/19 pass)**: 2 failures for toluene and NO2. These likely involve
   aromatic/charged molecule canonicalization differences.

4. **atomtype_parity (0/19 pass)**: rmgpu atom type assignments differ from RMG-Py. The
   current implementation assigns element symbols rather than RMG atom type labels.

5. **resonance_parity (17/19 pass)**: 2 failures match the SMILES parity failures.

6. **symmetry_parity (6/19 pass)**: rmgpu symmetry number calculation differs from RMG-Py.
   The simplified implementation does not match RMG-Py's algorithm.

## Reference Reads

- Read RMG-Py adjlist.py functions (from_adjacency_list, to_adjacency_list)
- Read RMG-Py molecule.py methods (to_adjacency_list, get_symmetry_number, generate_resonance_structures)

## Deviations

- Removed methoxy_radical [CO] from test set (invalid SMILES)
- No new reference code reads needed beyond what step file listed

## Commits

- `63ce210` job-01/step-07: Add job-01 gate with RMG-Py reference comparisons

## Next Step

The gate is RED for adjlist_parity, atomtype_parity, and symmetry_parity. These are HARD
failures per the gate definition. A fix-step is needed before job-01 can be marked done.

Recommended fix-step: `job-01-step-08-fix-parity.md` - implement proper adjlist formatting
parity and atom type assignments matching RMG-Py.
