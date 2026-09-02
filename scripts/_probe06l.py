"""Probe 12: exact bond types of the input ring + allyl candidate debug."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rdkit import Chem
from rdkit.Chem import BondType

mol = Molecule(smiles='C[CH]C1=CC=CC=C1')
m = mol._rdkit
print('=== input ring bonds (exact) ===')
# find the radical atom
rad = None
for a in m.GetAtoms():
    if a.GetNumRadicalElectrons() > 0:
        rad = a
print('radical atom idx:', rad.GetIdx(), 'aromatic:', rad.GetIsAromatic())
for b2 in rad.GetBonds():
    ipso_idx = b2.GetOtherAtomIdx(rad.GetIdx())
    ipso = m.GetAtomWithIdx(ipso_idx)
    print('  radical bonded to ipso idx %d (aromatic=%s)' % (ipso_idx, ipso.GetIsAromatic()))
    for bond in ipso.GetBonds():
        a1 = bond.GetBeginAtomIdx(); a2 = bond.GetEndAtomIdx()
        if a1 == rad.GetIdx() or a2 == rad.GetIdx():
            continue
        bt = bond.GetBondType()
        print('    bond %d-%d  GetBondType=%s  IsAromatic=%s  order=%.1f' % (
            a1, a2, bt, bond.GetIsAromatic(), bond.GetBondTypeAsDouble()))
