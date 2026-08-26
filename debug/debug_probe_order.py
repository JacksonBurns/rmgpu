#!/usr/bin/env python3
"""Probe RMG-Py vertex ordering: before/after to_smiles, input-order sensitivity."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/RMG-Py')
from rmgpy.molecule.molecule import Molecule as RMGMolecule

for smi in ['CC=O', 'O=CC', 'c1ccccc1', '[N+](=O)[O-]', 'O=C=O', 'CC(C)C']:
    mol = RMGMolecule(smiles=smi)
    before = [v.symbol for v in mol.vertices]
    adj = mol.to_adjacency_list()
    after_adj = [v.symbol for v in mol.vertices]
    sm = mol.to_smiles()
    after_sm = [v.symbol for v in mol.vertices]
    print(f"=== {smi}")
    print(f"  after init:      {before}")
    print(f"  after adjlist:   {after_adj}")
    print(f"  to_smiles() = {sm}")
    print(f"  after to_smiles: {after_sm}")
    first_lines = adj.splitlines()[:4]
    print(f"  adjlist head: {first_lines}")
    # lone pairs + radicals in original order
    print(f"  (lp,rad,charge) per vertex: {[(v.lone_pairs, v.radical_electrons, v.charge) for v in mol.vertices][:8]}")
