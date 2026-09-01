#!/usr/bin/env python3
"""Check whether RMG from_smiles produces explicit-H molecules and what
_match_reactant_to_template sees (vertex counts, atom types)."""
from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.molecule import Molecule

db = KineticsDatabase()
db.load_families(path=settings['database.directory'] + '/kinetics/families',
                 families=['H_Abstraction'])
fam = db.families['H_Abstraction']

for smi in ['C', '[H]', 'CC(=O)C']:
    m = Molecule().from_smiles(smi)
    print('SMILES', smi, 'n_vertices:', len(m.vertices),
          'symbols:', [a.element.symbol for a in m.vertices])

r = Molecule().from_smiles('C')
print('\nreactant C vertices:', len(r.vertices))
tmpl_reactants = [x.item for x in fam.forward_template.reactants]
print('fwd template n_slots:', len(tmpl_reactants))
for i, t in enumerate(tmpl_reactants):
    print('  slot', i, 'label', getattr(t, 'label', None),
          'n_vertices', len(t.vertices),
          'atomtypes', [[at.label for at in a.atomtype] for a in t.vertices][:4],
          'labels', [a.label for a in t.vertices][:6])
maps = fam._match_reactant_to_template(r, tmpl_reactants[0])
print('\nmatch(C, slot0): n_mappings', len(maps))
if maps:
    m0 = maps[0]
    for k, v in list(m0.items())[:6]:
        print('   react idx? atomtype', [at.label for at in k.atomtype],
              '-> tmpl label', v.label)
