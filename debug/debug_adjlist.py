#!/usr/bin/env python3
"""Debug script to check adjlist roundtrip."""
import sys
import os
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')

from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.adjlist import parse_adjlist, serialize_adjlist

# Test with ethane
smiles = "CC"
print(f"Testing: {smiles}")

# Create molecule
mol = Molecule(smiles=smiles)
print(f"Original SMILES: {mol.to_smiles()}")

# Get adjlist
adjlist_str = mol.to_adjlist()
print(f"Adjlist:\n{adjlist_str}\n")

# Try to parse it back
try:
    parsed_mol = Molecule.from_adjacency_list(adjlist_str)
    print(f"Parsed SMILES: {parsed_mol.to_smiles()}")
    print(f"Is isomorphic: {mol.is_isomorph(parsed_mol)}")
except Exception as e:
    print(f"Error parsing adjlist: {e}")
    import traceback
    traceback.print_exc()
