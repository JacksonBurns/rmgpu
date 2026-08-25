#!/usr/bin/env python3
"""Test resonance structure generation on sample molecules."""

from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import generate_resonance_structures

# Test molecules
test_molecules = [
    ("allyl radical", "[C]CC"),
    ("benzene", "c1ccccc1"),
    ("NO2", "O=[N+][O-]"),
    ("ethylene oxide", "O1CO1"),
    ("allyl cation", "C=C[CH2+]"),
    ("allyl anion", "C=C[CH2-]"),
]

for name, smiles in test_molecules:
    print(f"\n=== {name} ({smiles}) ===")
    try:
        mol = Molecule(smiles=smiles)
        resonance_structures = generate_resonance_structures(mol)
        print(f"  Input SMILES: {mol.to_smiles()}")
        print(f"  Number of resonance structures: {len(resonance_structures)}")
        for i, structure in enumerate(resonance_structures):
            print(f"  [{i}] {structure.to_smiles()}")
    except Exception as e:
        print(f"  Error: {e}")
