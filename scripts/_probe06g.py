"""Probe 7: confirm 5-form set, is_aromatic flags, structural distinctness."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import _generate_allyl_delocalization_resonance_structures
from rdkit import Chem

def skey(mol):
    """Structural fingerprint: per-atom (sym,rad,charge,sorted-neighbor(sym,order))."""
    em = mol._with_explicit_h()
    atoms = []
    for a in em.GetAtoms():
        nbrs = sorted((em.GetAtomWithIdx(b.GetOtherAtomIdx(a.GetIdx())).GetSymbol(),
                       round(b.GetBondTypeAsDouble(), 2), b.GetIsAromatic())
                      for b in a.GetBonds())
        atoms.append((a.GetSymbol(), a.GetNumRadicalElectrons(), a.GetFormalCharge(), tuple(nbrs)))
    return tuple(sorted(atoms))

def is_arom(mol):
    return any(b.GetIsAromatic() for b in mol._rdkit.GetBonds())

mol = Molecule(smiles='C[CH]C1=CC=CC=C1')
forms = [mol.copy()]
# BFS allyl
frontier = [mol._rdkit]
seen = {skey(mol)}
bfs = []
level = 0
while frontier and level < 4:
    level += 1
    nextf = []
    for rdmol in frontier:
        for s in _generate_allyl_delocalization_resonance_structures(rdmol):
            k = skey(s)
            if k not in seen:
                seen.add(k)
                bfs.append(s)
                nextf.append(s._rdkit)
    frontier = nextf
forms = [mol.copy()] + bfs
# also add aromatic form
from rmgpu.molecule.resonance import _generate_optimal_aromatic_resonance_structures
forms += _generate_optimal_aromatic_resonance_structures(mol._rdkit)

print('total forms (before filter):', len(forms))
for i, f in enumerate(forms):
    print('  [%d] %-24s is_arom=%s key_distinct' % (i, f.to_smiles(), is_arom(f)))
keys = [skey(f) for f in forms]
print('\ndistinct structural keys:', len(set(keys)), 'of', len(keys))
# find which are duplicates
from collections import Counter
for k, c in Counter(keys).items():
    if c > 1:
        idxs = [i for i, kk in enumerate(keys) if kk == k]
        print('  DUP group (x%d): forms %s' % (c, idxs))
