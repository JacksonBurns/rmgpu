#!/usr/bin/env python3
"""Settle rmgpu Molecule's internal H representation: explicit or implicit,
for SMILES-built and from_adjacency_list molecules. The group matcher must
match in the SAME vertex space the recipe engine applies recipes to."""
from rmgpu.molecule.molecule import Molecule

for smi in ['C', 'CC', '[H]', 'CC(=O)C', 'C1=CC=CC=C1']:
    m = Molecule(smiles=smi)
    atoms = m._rdkit.GetAtoms()
    nh = sum(1 for a in atoms if a.GetSymbol() == 'H')
    print('SMILES %-16r n_atoms=%d n_H=%d' % (smi, len(atoms), nh))

# from_adjacency_list
adj = "1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}\n2 H u0 p0 c0 {1,S}\n3 H u0 p0 c0 {1,S}\n4 H u0 p0 c0 {1,S}\n5 H u0 p0 c0 {1,S}\n"
m = Molecule.from_adjacency_list(adj)
atoms = m._rdkit.GetAtoms()
nh = sum(1 for a in atoms if a.GetSymbol() == 'H')
print('adjlist methane    n_atoms=%d n_H=%d' % (len(atoms), nh))
