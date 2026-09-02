"""Probe q: adjlist round-trip semantics for kek vs aromatic benzene forms.
Determines: (1) do to_adjlist strings differ between kek and aromatic reps?
(2) what does from_adjacency_list rebuild (S/D or AROMATIC)?
(3) what does filter_structures do to the full 5-form set?
(4) what does _allyl_bfs reach from the kek seed vs the aromatic seed?
"""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import (
    _allyl_bfs, _form_key, _form_is_aromatic, _has_standard_kekule_ring,
    _generate_optimal_aromatic_resonance_structures)
from rmgpu.molecule.resonance_filtration import filter_structures

kek = Molecule(smiles='C[CH]C1=CC=CC=C1')
aro = _generate_optimal_aromatic_resonance_structures(kek._rdkit)[0]

print('kek adjlist (ring part):')
print(kek.to_adjlist())
print()
print('aro adjlist (ring part):')
print(aro.to_adjlist())
print()
print('keys equal?', _form_key(kek) == _form_key(aro))

# round-trip each
for name, m in (('kek', kek), ('aro', aro)):
    rt = Molecule.from_adjacency_list(m.to_adjlist())
    em = rt._with_explicit_h()
    orders = []
    for b in em.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a1 < 6 and a2 < 6 and abs(a1 - a2) < 5:  # ring-ish
            orders.append((tuple(sorted((a1, a2))), b.GetBondTypeAsDouble(), b.GetIsAromatic()))
    print(f'{name} round-trip: smiles={rt.to_smiles()} is_arom={_form_is_aromatic(rt)}')
    print('   ring bonds:', sorted(orders))

# BFS from kek seed vs aro seed
print()
bfs_kek = _allyl_bfs([kek])
print('BFS from KEK seed ->', len(bfs_kek), 'new forms:')
for f in bfs_kek:
    print('   ', f.to_smiles(), 'is_arom=', _form_is_aromatic(f), 'std_kek=', _has_standard_kekule_ring(f))
bfs_aro = _allyl_bfs([aro])
print('BFS from ARO seed ->', len(bfs_aro), 'new forms:')
for f in bfs_aro:
    print('   ', f.to_smiles(), 'is_arom=', _form_is_aromatic(f), 'std_kek=', _has_standard_kekule_ring(f))

# filter_structures on the full set
all_forms = [kek] + [aro] + bfs_aro
print()
print('full set (dedup by key):')
seen = set()
deduped = []
for f in all_forms:
    k = _form_key(f)
    if k not in seen:
        seen.add(k)
        deduped.append(f)
print('  after key dedup:', len(deduped))
for f in deduped:
    print('   ', f.to_smiles())
fl = filter_structures(deduped, kek.copy())
print('after filter_structures:', len(fl))
for f in fl:
    print('   ', f.to_smiles(), 'is_arom=', _form_is_aromatic(f))
