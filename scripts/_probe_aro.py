"""Check: does form 0 (ar_rep) actually have 1.5 ring bonds after assign_fresh_ids,
and what are the Intra_ene template's bond orders?
"""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import enumeration as enum
from rmgpu.core.family import KineticsFamilies
from rmgpu.core.template import TemplateMatcher
from rmgpu.molecule.group import _explicit_graph

kf = KineticsFamilies().load('default')
fam = {f.label: f for f in kf.families}['Intra_ene_reaction']
matcher = TemplateMatcher(fam)

mol = Molecule(smiles='C[CH]C1=CC=CC=C1')
forms = enum.expand_resonance([mol])[0]
enum.assign_fresh_ids([forms])

print('== form 0 (ar_rep) ring bond orders after assign_fresh_ids ==')
f0 = forms[0]
r = f0._rdkit
print('  stored bonds (S/D/A):')
for b in r.GetBonds():
    if b.GetBondTypeAsDouble() in (1.0, 1.5, 2.0):
        print('    %d-%d o=%.1f aromatic=%s' % (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble(), b.GetIsAromatic()))
# explicit graph (what matcher sees)
g = _explicit_graph(f0)
orders = {}
for i in range(8):
    for (j, o) in g['adj'][i]:
        if j < 8:
            orders[(min(i,j), max(i,j))] = o
print('  matcher-view ring bond orders:', {k: v for k, v in sorted(orders.items()) if 1 < k[0] <= 7 and 1 < k[1] <= 7})

# Intra_ene template bond orders
print('\n== Intra_ene forward template [0] bond orders ==')
grp = fam.forward_template[0]
for (a, b), bond in grp.edges.items():
    print('  group bond %d-%d: orders=%s' % (a, b, bond.orders))
