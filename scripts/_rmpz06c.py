"""RMG-Py: per-form matching counts against the Intra_ene template."""
import sys
sys.path.insert(0, '/home/jackson/rmg/RMG-Py')
from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.molecule import Molecule

DBDIR = settings['database.directory']
m = Molecule()
m.from_smiles('C[CH]c1ccccc1')
m.update()
forms = m.generate_resonance_structures(keep_isomorphic=True)
for f in forms:
    f.update()

db = KineticsDatabase()
db.load_families(path=DBDIR + '/kinetics/families', families=['Intra_ene_reaction'])
fam = db.families['Intra_ene_reaction']
# find the forward template reactant group
print('forward_template.reactants:', len(fam.forward_template.reactants))
tgroup = fam.forward_template.reactants[0].item
if hasattr(tgroup, 'find_subgraph_isomorphisms'):
    grp = tgroup
else:
    grp = tgroup
print('template reactant group atoms:', len(grp.atoms) if hasattr(grp, 'atoms') else 'n/a')

print('\n=== per-form: template.find_subgraph_isomorphisms(form) ===')
for i, f in enumerate(forms):
    try:
        n = len(grp.find_subgraph_isomorphisms(f))
    except Exception as e:
        n = 'ERR %s' % e
    print('  form[%d] %s  reactive=%s  matchings=%s' % (
        i, f.to_smiles(), f.reactive, n))

# Also: the aromatic form vs R_Addition [D,T,B]
print('\n=== aromatic form vs R_Addition R_R [D,T,B] (benzene matching count) ===')
db2 = KineticsDatabase()
db2.load_families(path=DBDIR + '/kinetics/families', families=['R_Addition_MultipleBond'])
famn = db2.families['R_Addition_MultipleBond']
rr = famn.forward_template.reactants[0].item  # R_R
benz_ar = Molecule()
benz_ar.from_smiles('c1ccccc1')
benz_ar.update()
benz_kek = Molecule()
benz_kek.from_smiles('C1=CC=CC=C1')
benz_kek.update()
print('benz_ar is_aromatic:', benz_ar.is_aromatic(), 'benz_kek is_aromatic:', benz_kek.is_aromatic())
print('benz_ar bonds:', sorted(set(b.order for b in benz_ar.bonds)))
print('benz_kek bonds:', sorted(set(b.order for b in benz_kek.bonds)))
for label, b in [('ar', benz_ar), ('kek', benz_kek)]:
    try:
        n = len(rr.find_subgraph_isomorphisms(b))
    except Exception as e:
        n = 'ERR %s' % e
    print('  R_R vs benzene_%s: %s matchings' % (label, n))
