"""Diagnose: why does the kekulized benzene form render as c1ccccc1 (lowercase)?
Check bond flags + MolToSmiles for the kekulized form directly.
"""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import _generate_kekule_structure, _generate_optimal_aromatic_resonance_structures

# Kekulized form
kek = _generate_kekule_structure(Chem.Mol(Chem.MolFromSmiles('c1ccccc1')))[0]
print('== kekulized form (from _generate_kekule_structure) ==')
for b in kek._rdkit.GetBonds():
    if b.GetBondTypeAsDouble() in (1.0, 1.5, 2.0):
        print('  bond %d-%d order=%.1f aromatic=%s' % (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble(), b.GetIsAromatic()))
print('  stored _smiles:', kek._smiles)
print('  to_smiles():', kek.to_smiles())
print('  MolToSmiles(kek._rdkit):', Chem.MolToSmiles(kek._rdkit))
print('  MolToSmiles(canonical=True):', Chem.MolToSmiles(kek._rdkit, canonical=True))

# Now: what if I clear aromatic flags and re-sanitize?
rw = Chem.RWMol(kek._rdkit)
for b in rw.GetBonds():
    b.SetIsAromatic(False)
Chem.SanitizeMol(rw)
print('  after clearing aromatic flags + sanitize:')
print('    MolToSmiles:', Chem.MolToSmiles(Chem.Mol(rw)))

# What does the input molecule look like?
inp = Molecule(smiles='c1ccccc1')
print('\n== input Molecule(c1ccccc1) ==')
for b in inp._rdkit.GetBonds():
    if b.GetBondTypeAsDouble() in (1.0, 1.5, 2.0):
        print('  bond %d-%d order=%.1f aromatic=%s' % (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble(), b.GetIsAromatic()))
print('  to_smiles():', inp.to_smiles())

# Optimal aromatic form
opt = _generate_optimal_aromatic_resonance_structures(Chem.Mol(Chem.MolFromSmiles('c1ccccc1')))[0]
print('\n== optimal aromatic form ==')
print('  to_smiles():', opt.to_smiles())
for b in opt._rdkit.GetBonds():
    if b.GetBondTypeAsDouble() in (1.0, 1.5, 2.0):
        print('  bond %d-%d order=%.1f aromatic=%s' % (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble(), b.GetIsAromatic()))
