"""Probe 13: debug why single-step allyl gives 1 ortho not 2."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rdkit import Chem
from rdkit.Chem import BondType

mol = Molecule(smiles='C[CH]C1=CC=CC=C1')
m = mol._rdkit
rad_idx = None
for a in m.GetAtoms():
    if a.GetNumRadicalElectrons() > 0:
        rad_idx = a.GetIdx()
print('rad_idx:', rad_idx)

# Manually enumerate candidate bonds and test each shift
atom = m.GetAtomWithIdx(rad_idx)
candidates = {}
for b2 in atom.GetBonds():
    other_idx = b2.GetOtherAtomIdx(rad_idx)
    neighbor_atom = m.GetAtomWithIdx(other_idx)
    for bond in neighbor_atom.GetBonds():
        if bond.GetBeginAtomIdx() == rad_idx or bond.GetEndAtomIdx() == rad_idx:
            continue
        bt = bond.GetBondType()
        is_pi = bt in (BondType.DOUBLE, BondType.AROMATIC)
        is_ar = (not is_pi and bond.GetIsAromatic() and neighbor_atom.GetIsAromatic())
        if is_pi or is_ar:
            a1 = bond.GetBeginAtomIdx(); a2 = bond.GetEndAtomIdx()
            if other_idx in (a1, a2):
                key = frozenset((a1, a2))
                if key not in candidates:
                    candidates[key] = (a1, a2, 'pi' if is_pi else 'ar')
print('candidates:', candidates)

for key, (a1, a2, kind) in candidates.items():
    rad_bonded = None
    for b2 in atom.GetBonds():
        o = b2.GetOtherAtomIdx(rad_idx)
        if o in (a1, a2):
            rad_bonded = o
    if rad_bonded is None:
        print('  cand', a1, a2, ': no rad_bonded endpoint')
        continue
    target = a2 if rad_bonded == a1 else a1
    near = a1 if rad_bonded == a1 else a2
    new_mol = Chem.RWMol(m)
    new_mol.GetAtomWithIdx(target).SetNumRadicalElectrons(new_mol.GetAtomWithIdx(target).GetNumRadicalElectrons() + 1)
    new_mol.GetAtomWithIdx(rad_idx).SetNumRadicalElectrons(0)
    new_mol.GetBondBetweenAtoms(a1, a2).SetBondType(BondType.SINGLE)
    new_mol.GetBondBetweenAtoms(rad_idx, near).SetBondType(BondType.DOUBLE)
    for b in new_mol.GetBonds():
        b.SetIsAromatic(False)
    try:
        Chem.SanitizeMol(new_mol)
        Chem.Kekulize(new_mol, clearAromaticFlags=True)
        ok = True
    except Exception as e:
        ok = False
    print('  cand %d-%d (%s): rad->%d near=%d  kekulize=%s  %s' % (
        a1, a2, kind, target, near, ok,
        Chem.MolToSmiles(new_mol) if ok else str(e)))
