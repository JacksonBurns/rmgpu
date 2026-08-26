#!/usr/bin/env python3
"""Probe: does RDKit AddHs ordering match RMG-Py (OpenBabel) vertex order per test molecule?"""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/RMG-Py')
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu/gates')
from test_set import get_test_molecules, get_molecule_labels
from rmgpy.molecule.molecule import Molecule as RMGMolecule
from rdkit import Chem

def rdkit_order(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    return [a.GetSymbol() for a in mol.GetAtoms()]

for smi, label in zip(get_test_molecules(), get_molecule_labels()):
    rmg = RMGMolecule(smiles=smi)
    rmg_order = [v.symbol for v in rmg.vertices]
    rd_order = rdkit_order(smi)
    match = "MATCH" if rmg_order == rd_order else "DIFFER"
    print(f"{label:18s} {smi:14s} {match}")
    if rmg_order != rd_order:
        print(f"    rmgpy: {rmg_order}")
        print(f"    rdkit: {rd_order}")
