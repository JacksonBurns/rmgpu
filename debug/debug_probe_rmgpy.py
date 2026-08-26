#!/usr/bin/env python3
"""Probe RMG-Py behavior for the 19 gate test molecules (dev-only)."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/RMG-Py')
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu/gates')
from test_set import get_test_molecules, get_molecule_labels
from rmgpy.molecule.molecule import Molecule as RMGMolecule

smiles_list = get_test_molecules()
labels = get_molecule_labels()
assert len(smiles_list) == len(labels) == 19

for smi, label in zip(smiles_list, labels):
    mol = RMGMolecule(smiles=smi)
    print(f"===== {label} ({smi}) -> to_smiles() = {mol.to_smiles()}")
    print(f"  mult={mol.multiplicity}")
    # atom types in vertex order
    ats = [v.atomtype.label for v in mol.vertices]
    # symbols in vertex order
    syms = [v.symbol for v in mol.vertices]
    print(f"  vertex_symbols={syms}")
    print(f"  vertex_atomtypes={ats}")
    # check: do vertices include H?
    print(f"  n_vertices={len(mol.vertices)}")
    res = mol.generate_resonance_structures()
    print(f"  resonance={[s.to_smiles() for s in res]}")
    print(f"  symmetry={mol.get_symmetry_number()}")
