#!/usr/bin/env python3
"""rmgpu-side check: what does rmgpu currently produce (smiles, adjlist, atomtypes, symmetry, resonance)?"""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.atomtype import assign_atom_types
from rmgpu.molecule.symmetry import get_symmetry_number
from rmgpu.molecule.resonance import generate_resonance_structures

cases = [
    ("H2", "[H][H]"), ("O2", "[O][O]"), ("CH2", "[CH2]"), ("C2H2", "C#C"), ("N2", "[N][N]"),
    ("ethane", "CC"), ("methane", "C"), ("water", "O"), ("ethylene", "C=C"), ("isobutane", "CC(C)C"),
    ("methyl", "[CH3]"), ("formaldehyde", "C=O"), ("acetaldehyde", "CC=O"), ("CO2", "O=C=O"),
    ("benzene", "c1ccccc1"), ("toluene", "c1ccccc1C"), ("NO2", "[N+](=O)[O-]"),
    ("HNO3like", "[O-][N+](=O)"), ("glycolaldehyde", "OCC=O"),
]
for name, smi in cases:
    m = Molecule(smiles=smi)
    try:
        ats = assign_atom_types(m)
    except Exception as e:
        ats = f"ERR {e}"
    try:
        sym = get_symmetry_number(m)
    except Exception as e:
        sym = f"ERR {e}"
    try:
        res = [s.to_smiles() for s in generate_resonance_structures(m)]
    except Exception as e:
        res = f"ERR {e}"
    print(f"=== {name} ({smi})")
    print(f"  smiles={m.to_smiles()}  sym={sym}")
    print(f"  atomtypes={ats}")
    print(f"  resonance={res}")
