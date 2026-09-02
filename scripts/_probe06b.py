"""Probe 2: adjlist round-trip aromaticity + RMG-Py ground truth."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rdkit import Chem

# Aromatic form of 1-phenylethyl
ar = Molecule(smiles='C[CH]c1ccccc1')
print('aromatic form smiles:', ar.to_smiles())
print('  has aromatic bonds:', any(b.GetIsAromatic() for b in ar._rdkit.GetBonds()))
print('  to_adjlist() =')
al = ar.to_adjlist()
print(al)
print('  round-trip from_adjacency_list:')
rt = Molecule.from_adjacency_list(al)
for b in rt._rdkit.GetBonds():
    o = b.GetBondTypeAsDouble()
    if o == 1.5:
        print('    AROMATIC bond', o)
    elif o in (1, 2, 3):
        pass
ar_bonds = [b.GetBondTypeAsDouble() for b in rt._rdkit.GetBonds()]
print('  round-trip bond orders:', sorted(set(ar_bonds)))
print('  round-trip smiles:', rt.to_smiles())
