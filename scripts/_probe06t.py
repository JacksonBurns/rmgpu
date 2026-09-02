"""Probe t: THE decisive experiment.
Build the 5-form benzylic set (kek seed BFS), run the REAL assign_fresh_ids
round-trip, and dump exactly what _explicit_graph (the matcher input) sees
for each form: bond orders + aromatic flags on ring bonds. This determines
the exact scope of fix 3.
"""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import (
    _allyl_bfs, _form_key, _generate_optimal_aromatic_resonance_structures,
    _generate_kekule_structure)
from rmgpu.molecule.group import _explicit_graph
from rmgpu.core.enumeration import assign_fresh_ids

mol = Molecule(smiles='C[CH]C1=CC=CC=C1')

# Build the intended 5-form set
cands = [mol.copy()]
cands.extend(_allyl_bfs([mol.copy()]))
opt = _generate_optimal_aromatic_resonance_structures(mol._rdkit)
cands.extend(opt)
cands.extend(_generate_kekule_structure(mol._rdkit))
seen = set()
forms = []
for f in cands:
    k = _form_key(f)
    if k not in seen:
        seen.add(k)
        forms.append(f)
print('5-form set (structural key dedup):', len(forms))
for i, f in enumerate(forms):
    r = f._rdkit
    print('  [%d] %s' % (i, f.to_smiles()))

# PRE-round-trip: ring bond orders in _explicit_graph
print('\nPRE-round-trip matcher view (AddHs):')
for i, f in enumerate(forms):
    g = _explicit_graph(f)
    # find 6-ring-ish heavy atom bonds among first 8 atoms
    orders = {}
    for a, nbrs in enumerate(g['adj'][:8]):
        for (b, o) in nbrs:
            if b < 8:
                key = (min(a, b), max(a, b))
                orders[key] = o
    ringish = {k: v for k, v in orders.items() if 1 < k[0] <= 7 and 1 < k[1] <= 7}
    print('  [%d] ring bond orders: %s' % (i, sorted(ringish.items())))

# Run the real round-trip
species = [list(forms)]
assign_fresh_ids(species)
forms2 = species[0]
print('\nPOST-round-trip (assign_fresh_ids) matcher view:')
for i, f in enumerate(forms2):
    r = f._rdkit
    em = f._with_explicit_h()
    ri = em.GetRingInfo()
    rings = [tuple(x) for x in ri.AtomRings()]
    g = _explicit_graph(f)
    orders = {}
    for a, nbrs in enumerate(g['adj'][:8]):
        for (b, o) in nbrs:
            if b < 8:
                key = (min(a, b), max(a, b))
                orders[key] = o
    ringish = {k: v for k, v in orders.items() if 1 < k[0] <= 7 and 1 < k[1] <= 7}
    # stored bond aromatic flags
    flags = {}
    for b in r.GetBonds():
        if b.GetBondTypeAsDouble() in (1.0, 2.0):
            key = (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            flags[key] = (round(b.GetBondTypeAsDouble(), 1), b.GetIsAromatic())
    print('  [%d] smiles=%s' % (i, f.to_smiles()[:40]))
    print('       rings in AddHs: %s' % (rings if rings else 'NONE'))
    print('       matcher ring orders: %s' % sorted(ringish.items()))
    print('       stored S/D (order,aromatic): %s' % {k: v for k, v in flags.items() if 1 < k[0] <= 7 and 1 < k[1] <= 7})
    print('       props: arom_rep=%s reactive=%s' % (
        r.GetProp('rmgpu_aromatic_rep') if r.HasProp('rmgpu_aromatic_rep') else '-',
        r.GetProp('rmgpu_reactive') if r.HasProp('rmgpu_reactive') else '-'))
