"""Test: force the kekule form to true S/D + cleared aromatic flags -> uppercase SMILES."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rdkit.Chem import BondType
from rmgpu.molecule.molecule import Molecule

def force_kekule(mol):
    """Set alternating S/D on 6-membered rings, clear aromatic flags, sanitize."""
    rw = Chem.RWMol(mol)
    for ring in rw.GetRingInfo().AtomRings():
        if len(ring) == 6:
            for k in range(6):
                a, b = ring[k], ring[(k + 1) % 6]
                bd = rw.GetBondBetweenAtoms(a, b)
                bt = bd.GetBondTypeAsDouble()
                # alternate: keep as-is if already S/D, else set based on position
                if abs(bt - 1.5) < 0.1:
                    bt = 2.0  # aromatic -> start as double
                # just keep whatever S/D it is; we only need to clear aromatic
    for b in rw.GetBonds():
        b.SetIsAromatic(False)
    # Sanitize to refresh
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        pass
    return Chem.Mol(rw)

# Kekulized benzene (from _generate_kekule_structure)
from rmgpu.molecule.resonance import _generate_kekule_structure
kek = _generate_kekule_structure(Chem.Mol(Chem.MolFromSmiles('c1ccccc1')))[0]
print('before: to_smiles=', kek.to_smiles())
fixed = force_kekule(kek._rdkit)
print('after force_kekule: MolToSmiles=', Chem.MolToSmiles(fixed))
m = Molecule._from_rdmol(fixed)
print('  Molecule to_smiles=', m.to_smiles())

# Toluene
ket = _generate_kekule_structure(Chem.Mol(Chem.MolFromSmiles('Cc1ccccc1')))[0]
print('\ntoluene before: to_smiles=', ket.to_smiles())
fixedt = force_kekule(ket._rdkit)
print('toluene after: MolToSmiles=', Chem.MolToSmiles(fixedt))
mt = Molecule._from_rdmol(fixedt)
print('  Molecule to_smiles=', mt.to_smiles())
