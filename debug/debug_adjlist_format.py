#!/usr/bin/env python3
"""Debug script to compare adjlist formatting with RMG-Py."""
import sys
import os
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')

from rmgpu.molecule.molecule import Molecule

# Test with ethane
smiles = "CC"
print(f"Testing: {smiles}\n")

# Create molecule
mol = Molecule(smiles=smiles)
print(f"Original SMILES: {mol.to_smiles()}")

# Get adjlist
adjlist_str = mol.to_adjlist()
print(f"RMGPU adjlist:\n{repr(adjlist_str)}\n")

# Expected RMG-Py adjlist (from reference)
# For ethane (CC), RMG-Py produces:
# multiplicity 1
#  1    C   u0p0 { 2,S}
#  2    C   u0p0 { 1,S}
#  3    H   u0p0 { 1,S}
#  4    H   u0p0 { 1,S}
#  5    H   u0p0 { 1,S}
#  6    H   u0p0 { 2,S}
#  7    H   u0p0 { 2,S}
#  8    H   u0p0 { 2,S}

print("\nExpected RMG-Py adjlist (approximate):")
expected = """
 1    C   u0p0 { 2,S}
 2    C   u0p0 { 1,S}
 3    H   u0p0 { 1,S}
 4    H   u0p0 { 1,S}
 5    H   u0p0 { 1,S}
 6    H   u0p0 { 2,S}
 7    H   u0p0 { 2,S}
 8    H   u0p0 { 2,S}
"""
print(expected)

# Compare line by line
rmgpu_lines = adjlist_str.strip().split('\n')
expected_lines = expected.strip().split('\n')

print(f"Number of lines: rmgpu={len(rmgpu_lines)}, expected={len(expected_lines)}")
for i, (r, e) in enumerate(zip(rmgpu_lines, expected_lines)):
    match = r.strip() == e.strip()
    print(f"  Line {i+1}: {'MATCH' if match else 'DIFFER'}")
    print(f"    rmgpu: {repr(r)}")
    print(f"    expected: {repr(e)}")
