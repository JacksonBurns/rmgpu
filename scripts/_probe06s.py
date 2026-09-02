"""Probe s: DEFINITIVE bond-level ground truth.
For each form, dump bond orders + aromatic flags:
  (a) as stored (mol._rdkit),
  (b) after AddHs (what _form_is_aromatic / matcher use),
  (c) after the assign_fresh_ids adjlist round-trip (what the matcher matches).
Also: does AddHs re-aromatize a kekulized ring?
"""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
import rmgpu.molecule.resonance as R


def ring_bonds(mol, label):
    """Return sorted list of (a,b,order,aromatic) for the 6-ring bonds of mol."""
    em = mol._with_explicit_h()
    ri = em.GetRingInfo()
    rings = [r for r in ri.AtomRings() if len(r) == 6]
    out = []
    if rings:
        ring = rings[0]
        for k in range(6):
            a, b = ring[k], ring[(k + 1) % 6]
            bd = em.GetBondBetweenAtoms(a, b)
            out.append((min(a, b), max(a, b), round(bd.GetBondTypeAsDouble(), 2), bd.GetIsAromatic()))
    return sorted(out)


def stored_bonds(mol):
    """Bond orders+aromatic flags on the stored (implicit-H) mol."""
    out = []
    for b in mol._rdkit.GetBonds():
        o = b.GetBondTypeAsDouble()
        if o in (1.0, 1.5, 2.0):
            out.append((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                        max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                        round(o, 2), b.GetIsAromatic()))
    return sorted(out)


# 1. Kekulized input
kek = Molecule(smiles='C[CH]C1=CC=CC=C1')
print('=== kek input C[CH]C1=CC=CC=C1 ===')
print('stored S/D/A bonds :', stored_bonds(kek))
print('AddHs ring bonds    :', ring_bonds(kek, 'addh'))
print('_form_is_aromatic   :', R._form_is_aromatic(kek))

# 2. After adjlist round-trip (what assign_fresh_ids produces)
kek_rt = Molecule.from_adjacency_list(kek.to_adjlist())
print('--- after adjlist round-trip ---')
print('to_smiles           :', kek_rt.to_smiles())
print('stored S/D/A bonds :', stored_bonds(kek_rt))
print('AddHs ring bonds    :', ring_bonds(kek_rt, 'addh'))
print('_form_is_aromatic   :', R._form_is_aromatic(kek_rt))
print('_has_std_kekule_ring:', R._has_standard_kekule_ring(kek_rt))

# 3. Pure aromatic benzene
benz_ar = Molecule._from_rdmol(Chem.Mol(Chem.MolFromSmiles('c1ccccc1')))
print('\n=== aromatic benzene c1ccccc1 ===')
print('stored S/D/A bonds :', stored_bonds(benz_ar))
print('AddHs ring bonds    :', ring_bonds(benz_ar, 'addh'))
print('_form_is_aromatic   :', R._form_is_aromatic(benz_ar))
benz_ar_rt = Molecule.from_adjacency_list(benz_ar.to_adjlist())
print('--- after adjlist round-trip ---')
print('to_smiles           :', benz_ar_rt.to_smiles())
print('stored S/D/A bonds :', stored_bonds(benz_ar_rt))
print('_form_is_aromatic   :', R._form_is_aromatic(benz_ar_rt))

# 4. Kekulized benzene (explicit S/D, no aromatic flag) - force it
benz_k = Chem.RWMol(Chem.Mol(Chem.MolFromSmiles('C1=CC=CC=C1')))
print('\n=== does MolFromSmiles("C1=CC=CC=C1") stay kekulized? ===')
print('stored bonds       :', stored_bonds(Molecule._from_rdmol(Chem.Mol(Chem.MolFromSmiles('C1=CC=CC=C1')))))
print('is_aromatic(flag)  :', R._is_aromatic(benz_k))

# 5. AddHs re-aromatize test: take a genuinely kekulized mol (clear flags), AddHs
benz_k2 = Chem.Mol(Chem.MolFromSmiles('C1=CC=CC=C1'))
for b in benz_k2.GetBonds():
    b.SetIsAromatic(False)
Chem.SanitizeMol(benz_k2)
mk = Molecule._from_rdmol(benz_k2)
print('\n=== forced-kekulized benzene (flags cleared) ===')
print('stored bonds       :', stored_bonds(mk))
print('AddHs ring bonds    :', ring_bonds(mk, 'addh'))
print('_form_is_aromatic   :', R._form_is_aromatic(mk))
