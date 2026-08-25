# job-01/step-02a: Molecule wrapper: construction and properties

## What was built

- **rmgpu/molecule/molecule.py**: `Molecule` class wrapping an RDKit `RWMol` in canonical kekulized form. Provides construction from SMILES or InChI (InChI takes precedence when both given), `to_smiles()`, `to_inchi()`, `get_formula()` (Hill order), `get_charge()` (sum of formal charges), `get_radical_count()` (sum of unpaired electrons), and equality/hashing based on canonical SMILES.
- **tests/test_molecule.py**: 29 tests covering construction from SMILES and InChI, formula, charge/radical accessors, equality and hashing across equivalent structures and different atom orders, and SMILES output matching RDKit canonical form.

## Checks run

Command: `pytest tests/test_molecule.py -v`

Result: **29 passed**. All construction, formula, charge, radical count, SMILES, InChI, equality, and hashing tests pass.

## Reference reads beyond the list

- **RMG-Py/rmgpy/molecule/element.py**: Read to understand the element table and periodic system data. This was outside the explicitly listed reference set but was referenced in the step file as a direct dependency to port as data. The element table itself was not ported in this step since it is not directly needed for the construction and property layer being built.

## Deviations from the step file

- **InChI priority**: The step file mentioned that InChI takes precedence when both SMILES and InChI are provided, matching RMG-Py behavior. This was implemented as specified.
- **Adjacency-list construction**: Not implemented in this step as the step file indicated this would be added in the adjlist step.

## What the next step should know first

- The `Molecule` class is ready for labeled-atom and bond-label APIs in step-02b.
- Equality and hashing are based on canonical SMILES, which is stable under atom reordering and equivalent structural representations.
- The molecule is stored in canonical kekulized form internally.
