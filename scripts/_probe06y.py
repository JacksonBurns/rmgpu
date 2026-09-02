"""Probe y: raw bond orders of each generated resonance form (stored + AddHs)."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import (
    _allyl_bfs, _generate_optimal_aromatic_resonance_structures,
    _generate_kekule_structure, _form_key, _aromatic_rep, _form_is_aromatic,
    _has_standard_kekule_ring)


def raw_orders(m):
    out = []
    for b in m._rdkit.GetBonds():
        o = b.GetBondTypeAsDouble()
        if o in (1.0, 1.5, 2.0):
            out.append((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                        max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                        round(o, 2), b.GetIsAromatic()))
    return sorted(out)


def addh_orders(m):
    em = m._with_explicit_h()
    out = []
    for b in em.GetBonds():
        o = b.GetBondTypeAsDouble()
        if o in (1.0, 1.5, 2.0):
            out.append((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                        max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                        round(o, 2), b.GetIsAromatic()))
    return sorted(out)


mol = Molecule(smiles='C[CH]C1=CC=CC=C1')
print('input raw:', raw_orders(mol))
print('input addh:', addh_orders(mol))
print()

print('== optimal aromatic (SetAromaticity path) ==')
opt = _generate_optimal_aromatic_resonance_structures(mol._rdkit)
for o in opt:
    print(' smiles:', o.to_smiles())
    print('  raw :', raw_orders(o))
    print('  addh:', addh_orders(o))
    print('  is_arom:', _form_is_aromatic(o), 'ar_rep:', _aromatic_rep(o))

print()
print('== kekule structure ==')
kek = _generate_kekule_structure(mol._rdkit)
for o in kek:
    print(' smiles:', o.to_smiles())
    print('  raw :', raw_orders(o))
    print('  addh:', addh_orders(o))
    print('  is_arom:', _form_is_aromatic(o), 'std_kek:', _has_standard_kekule_ring(o))

print()
print('== allyl bfs from kekulized ==')
bfs = _allyl_bfs([mol.copy()])
for o in bfs:
    print(' smiles:', o.to_smiles(), ' raw:', raw_orders(o), ' addh:', addh_orders(o))
