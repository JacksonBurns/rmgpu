"""Pin down WHY the kekule form dedupes with the aromatic for benzene/benzylic.
Compare stored-mol bond orders + aromatic flags + _form_key for input vs kekule.
"""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import (
    _generate_kekule_structure, _generate_optimal_aromatic_resonance_structures,
    _form_key, _form_is_aromatic, _has_standard_kekule_ring)


def desc(m, label):
    r = m._rdkit
    bonds = []
    for b in r.GetBonds():
        o = b.GetBondTypeAsDouble()
        if o in (1.0, 1.5, 2.0):
            bonds.append((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                          max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                          round(o, 2), b.GetIsAromatic()))
    print('  %-16s to_smiles=%-16s is_arom=%s std_kek=%s' % (
        label, m.to_smiles(), _form_is_aromatic(m), _has_standard_kekule_ring(m)))
    print('      bonds:', sorted(bonds))
    print('      form_key[:3]:', str(_form_key(m))[:90])
    return _form_key(m)


print('=== BENZENE ===')
inp = Molecule(smiles='c1ccccc1')
k1 = desc(inp, 'input')
opt = _generate_optimal_aromatic_resonance_structures(inp._rdkit)
kek = _generate_kekule_structure(inp._rdkit)
k2 = desc(opt[0], 'optimal-arom')
k3 = desc(kek[0], 'kekule')
print('  key(input)==key(optimal)?', k1 == k2)
print('  key(input)==key(kekule)?', k1 == k3)
print('  key(optimal)==key(kekule)?', k2 == k3)

print('\n=== BENZYLIC ===')
inp2 = Molecule(smiles='C[CH]C1=CC=CC=C1')
b1 = desc(inp2, 'input')
opt2 = _generate_optimal_aromatic_resonance_structures(inp2._rdkit)
kek2 = _generate_kekule_structure(inp2._rdkit)
b2 = desc(opt2[0], 'optimal-arom')
b3 = desc(kek2[0], 'kekule')
print('  key(input)==key(optimal)?', b1 == b2)
print('  key(input)==key(kekule)?', b1 == b3)
print('  key(optimal)==key(kekule)?', b2 == b3)
