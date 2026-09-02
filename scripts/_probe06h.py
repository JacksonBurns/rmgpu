"""Probe 8: verify the 1.5-aromatic-form mechanism + BFS form set."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rdkit import Chem

# 1. Does MolFromSmiles aromatic give AROMATIC bond type + 1.5?
ar = Chem.MolFromSmiles('C[CH]c1ccccc1')
ring = []
for b in ar.GetBonds():
    if b.GetIsAromatic():
        ring.append((b.GetBondType(), b.GetBondTypeAsDouble()))
print('MolFromSmiles aromatic ring bonds:', set(ring))

# 2. to_adjlist on a true-aromatic form
ar_mol = Molecule._from_rdmol(Chem.Mol(ar))
print('ar_mol to_smiles:', ar_mol.to_smiles())
al = ar_mol.to_adjlist()
print('to_adjlist has B(1.5) token:', ',B}' in al.replace(' ', '') or 'B}' in al.replace(' ', ''))
# print ring bond tokens
for line in al.splitlines():
    if 'C' in line and ('{3,' in line or '{4,' in line or '{5,' in line or '{6,' in line or '{7,' in line or '{8,' in line):
        pass
print('--- adjlist (ring lines) ---')
for line in al.splitlines():
    print('  ', line)

# 3. round-trip preserves 1.5?
rt = Molecule.from_adjacency_list(al)
em = rt._with_explicit_h()
print('round-trip bond orders:', sorted(set(b.GetBondTypeAsDouble() for b in em.GetBonds())))

# 4. matcher _explicit_graph sees 1.5?
from rmgpu.molecule.group import _explicit_graph
g = _explicit_graph(rt)
bo = set()
for adj in g['adj']:
    for (j, o) in adj:
        bo.add(round(o, 2))
print('_explicit_graph bond orders:', sorted(bo))

# 5. Compare: kekulized input form to_adjlist
kek = Molecule(smiles='C[CH]C1=CC=CC=C1')
alk = kek.to_adjlist()
print('kekulized input to_adjlist has B token:', ',B}' in alk.replace(' ', '') or 'B}' in alk.replace(' ', ''))
gtk = _explicit_graph(kek)
bok = set()
for adj in gtk['adj']:
    for (j, o) in adj:
        bok.add(round(o, 2))
print('kekulized input _explicit_graph bond orders:', sorted(bok))
