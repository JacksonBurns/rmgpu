"""Probe w: LINCHPIN - does the matcher (_explicit_graph, via AddHs) read 1.5
for a re-aromatized (AROMATIC bond type) benzene ring? If AddHs re-kekulizes,
re-aromatization is useless and fix 3 must work differently.
"""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rdkit.Chem import BondType
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.group import _explicit_graph


def ring_orders_via_explicit_graph(mol):
    g = _explicit_graph(mol)
    out = []
    n = len(g['atoms'])
    for i in range(min(n, 8)):
        for (j, o) in g['adj'][i]:
            if j < 8:
                out.append((min(i, j), max(i, j), round(o, 2)))
    return sorted(set(out))


# Build a re-aromatized benzene: start kekulized, set ring bonds AROMATIC
kek = Chem.RWMol(Chem.Mol(Chem.MolFromSmiles('C1=CC=CC=C1')))
# find the 6 ring bonds
ri = kek.GetRingInfo()
ring = [r for r in ri.AtomRings() if len(r) == 6][0]
for k in range(6):
    a, b = ring[k], ring[(k + 1) % 6]
    kek.GetBondBetweenAtoms(a, b).SetBondType(BondType.AROMATIC)
for b in kek.GetBonds():
    if b.GetBondType() != BondType.AROMATIC:
        b.SetIsAromatic(False)
Chem.SanitizeMol(kek)
rearo = Molecule._from_rdmol(Chem.Mol(kek))
print('re-aromatized benzene to_smiles:', rearo.to_smiles())

# stored bonds
print('stored bond orders:', sorted(set(round(b.GetBondTypeAsDouble(), 2) for b in rearo._rdkit.GetBonds())))

# AddHs bond orders (what _explicit_graph uses)
em = rearo._with_explicit_h()
print('AddHs bond orders:', sorted(set(round(b.GetBondTypeAsDouble(), 2) for b in em.GetBonds())))
print('AddHs aromatic flags:', [b.GetIsAromatic() for b in em.GetBonds() if b.GetBondTypeAsDouble() in (1, 2, 1.5)])

# the matcher view
print('matcher (_explicit_graph) ring orders:', ring_orders_via_explicit_graph(rearo))

# Now the SAME molecule but WITHOUT re-aromatization (kekulized S/D)
kek2 = Molecule._from_rdmol(Chem.Mol(Chem.MolFromSmiles('C1=CC=CC=C1')))
print('\nkekulized benzene matcher ring orders:', ring_orders_via_explicit_graph(kek2))
