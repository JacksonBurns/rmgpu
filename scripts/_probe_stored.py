"""Probe: STORED mol (m._rdkit, no AddHs) bond flags at generation time,
for the benzylic 5-form set and benzene. This is what the structural key
must be computed on (AddHs re-aromatizes kekulized benzene, destroying the
distinction).
"""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import (
    _allyl_bfs, _generate_optimal_aromatic_resonance_structures,
    _generate_kekule_structure)


def stored_ring_flags(m):
    """On the STORED mol (no AddHs): for the first 6-ring, list
    (order, is_aromatic) per bond."""
    r = m._rdkit
    out = []
    for ring in r.GetRingInfo().AtomRings():
        if len(ring) == 6:
            for k in range(6):
                a, b = ring[k], ring[(k + 1) % 6]
                bd = r.GetBondBetweenAtoms(a, b)
                out.append((round(bd.GetBondTypeAsDouble(), 2), bd.GetIsAromatic()))
            break
    # radical position (stored idx)
    rads = [a.GetIdx() for a in r.GetAtoms() if a.GetNumRadicalElectrons() > 0]
    return sorted(out), rads


mol = Molecule(smiles='C[CH]C1=CC=CC=C1')
cands = [mol.copy()]
cands.extend(_allyl_bfs([mol.copy()]))
cands.extend(_generate_optimal_aromatic_resonance_structures(mol._rdkit))
cands.extend(_generate_kekule_structure(mol._rdkit))

def sig_stored(m):
    """Proposed key: stored mol, aromatic-flag->1.5, heavy-atom bonds + radicals."""
    r = m._rdkit
    bonds = []
    for b in r.GetBonds():
        o = b.GetBondTypeAsDouble()
        if b.GetIsAromatic():
            o = 1.5
        if o in (1.0, 1.5, 2.0, 3.0):
            bonds.append((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                          max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), o))
    atoms = [(a.GetIdx(), a.GetSymbol(), a.GetNumRadicalElectrons(), a.GetFormalCharge())
             for a in r.GetAtoms()]
    return (tuple(sorted(bonds)), tuple(atoms))

seen, forms = set(), []
print('== benzylic, STORED-mol key dedup ==')
for f in cands:
    s = sig_stored(f)
    if s not in seen:
        seen.add(s)
        forms.append(f)
for i, f in enumerate(forms):
    flags, rads = stored_ring_flags(f)
    print('  [%d] %-22s ring=%s rad=%s' % (i, f.to_smiles(), flags, rads))
print('  total distinct:', len(forms))

print('\n== benzene, STORED-mol key dedup ==')
bmol = Molecule(smiles='c1ccccc1')
bcands = [bmol.copy()]
bcands.extend(_generate_optimal_aromatic_resonance_structures(bmol._rdkit))
bcands.extend(_generate_kekule_structure(bmol._rdkit))
bseen, bforms = set(), []
for f in bcands:
    s = sig_stored(f)
    if s not in bseen:
        bseen.add(s)
        bforms.append(f)
for i, f in enumerate(bforms):
    flags, rads = stored_ring_flags(f)
    print('  [%d] %-16s ring=%s to_smiles=%s' % (i, f.to_smiles(), flags, f.to_smiles()))
print('  total distinct:', len(bforms))
