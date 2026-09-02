"""Probe 9: exact bond type per form after assign_fresh_ids (the crux)."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import enumeration as enum
from rmgpu.molecule.group import _explicit_graph

mol = Molecule(smiles='C[CH]C1=CC=CC=C1')
species = enum.expand_resonance([mol])
print('raw forms: %d' % len(species[0]))
enum.assign_fresh_ids(species)
print('after assign_fresh_ids: %d forms' % len(species[0]))
for i, f in enumerate(species[0]):
    # bond types in the STORED rdmol (pre explicit-H)
    stored = sorted(set(round(b.GetBondTypeAsDouble(), 2) for b in f._rdkit.GetBonds()))
    narom = sum(1 for b in f._rdkit.GetBonds() if b.GetIsAromatic())
    # _explicit_graph bond orders (what the matcher sees)
    g = _explicit_graph(f)
    eb = set()
    for adj in g['adj']:
        for (j, o) in adj:
            eb.add(round(o, 2))
    # ring 6 pattern
    from rdkit import Chem
    em = f._with_explicit_h()
    ring_pat = ''
    for ring in em.GetRingInfo().AtomRings():
        if len(ring) == 6:
            pat = []
            for k in range(6):
                o = em.GetBondBetweenAtoms(ring[k], ring[(k+1)%6]).GetBondTypeAsDouble()
                pat.append('D' if abs(o-2)<.1 else 'S' if abs(o-1)<.1 else '?')
            ring_pat = ''.join(pat)
    print('  form[%d] %s  stored_orders=%s n_arom_bonds=%d explicit_graph_orders=%s ring6=%s' % (
        i, f.to_smiles()[:30], stored, narom, sorted(eb), ring_pat))
