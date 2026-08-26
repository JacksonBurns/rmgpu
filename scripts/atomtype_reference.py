#!/usr/bin/env python
"""
Reference script to generate atom type assignments using RMG-Py.

Usage:
    /home/jackson/miniforge3/envs/rmg_env/bin/python atomtype_reference.py > gates/baselines/atomtypes/reference.txt
"""

import sys
import os

# Add RMG-Py to path
sys.path.insert(0, '/home/jackson/rmgpu/RMG-Py')

from rmgpy.molecule.molecule import Molecule
from rmgpy.molecule.atomtype import assign_atom_types

# Test molecules - same as in test_atomtype.py
TEST_MOLECULES = [
    'C',           # methane
    'CC',          # ethane
    'C=C',         # ethylene
    'C#C',         # acetylene
    'CCO',         # ethanol
    'CC(=O)C',     # acetone
    'c1ccccc1',    # benzene
    'N',           # ammonia
    'N=N',         # azo compound
    'N#N',         # dinitrogen
    'O',           # water
    'O=C=O',       # carbon dioxide
    'C1CCCCC1',    # cyclohexane
    'CCN',         # ethylamine
    'CC(=O)O',     # acetic acid
    'CCSCC',       # 1,2-dimethylsulfide
    'CCC',         # propane
    'C(=O)C',      # formaldehyde
    'CC(F)(F)F',   # fluoroethane
    'CCCl',        # chloroethane
    'CCBr',        # bromoethane
    'CC#N',        # acetonitrile
    'CC(=O)N',     # acetamide
    'CCS',         # methyl sulfide
    'C=C(C)C',     # isobutylene
    'CC(C)(C)C',   # neopentane
    'C1=C2C(=C1)C=CC=C2',  # naphthalene
    'C1=CC=NC=N1',   # pyrazine
    'CCOC',        # diethyl ether
    'CC(C)O',      # isopropanol
]

def main():
    print("SMILESType1Type2Type3...")
    for smiles in TEST_MOLECULES:
        try:
            mol = Molecule(smiles=smiles)
            mol.update_atomtypes()
            # Get atom type labels
            types = [atom.atomtype.label for atom in mol.atoms]
            types_str = ','.join(types)
            print(f"{smiles},{types_str}")
        except Exception as e:
            print(f"Error for {smiles}: {e}")

if __name__ == '__main__':
    main()