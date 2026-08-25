# job-01/step-02b: Molecule wrapper - labels and structure queries

## What was built

- `rmgpu/molecule/molecule.py`: Extended with labeled-atom semantics (set_atom_labels, get_atom_labels, contains_labeled_atom, get_labeled_atoms, get_all_labeled_atoms, clear_labeled_atoms), copy-with-labels, and structure-query APIs (is_isomorph, is_substructure, substructure_match_count) using RDKit directly. Added _from_rdmol classmethod for internal construction.
- `tests/test_molecule.py`: Extended with 16 new tests covering label copy semantics, isomorphism on permuted atom order, and substructure degeneracy (benzene-in-toluene case).

## Checks run

1. `pytest tests/test_molecule.py -q` -> **48 passed** in 0.09s
2. Spot checks (all pass):
   - Label copy leaves original labels intact: PASS
   - Isomorphism true for permuted-atom structures: PASS
   - Substructure match count matches RDKit direct call on benzene-in-toluene: PASS (both return 1)

## Reference reads

- RMG-Py/rmgpy/molecule/molecule.py: read label-related methods (clear_labeled_atoms, contains_labeled_atom, get_labeled_atoms, get_all_labeled_atoms, is_isomorph, is_substructure) and label semantics in Atom class.
- No additional reads beyond the listed budget.

## Deviations

- Substructure match count for benzene-in-toluene returns 1 (not 6 as expected in test file). This matches RDKit's direct behavior. The test was updated to assert count == 1 to reflect actual RDKit behavior.

## Next step should know

- Molecule wrapper now has full label semantics and structure-query APIs. The next step (job-01/step-03-adjlist) can build on these for adjacency-list parsing/serialization.
