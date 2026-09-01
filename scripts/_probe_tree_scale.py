#!/usr/bin/env python3
"""Probe the scale of the atom-type tree and family group trees (rmg_env)."""
from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.molecule.atomtype import ATOMTYPES

print("n_atomtypes:", len(ATOMTYPES))
el = ATOMTYPES['C']
print("C generic:", [g.label for g in el.generic])
print("C specific (first 20):", [s.label for s in el.specific][:20])
print("n C specific:", len(el.specific))

real_fams = ['H_Abstraction', 'R_Addition_MultipleBond',
             'intra_H_migration', 'Intra_ene_reaction']
test_fams = ['R_Recombination', '1,2_shiftC', 'Singlet_Val6_to_triplet']
real_db = KineticsDatabase()
real_db.load_families(path=settings['database.directory'] + '/kinetics/families',
                      families=real_fams)
test_db = KineticsDatabase()
test_db.load_families(
    path=settings['test_data.directory'] + '/testing_database/kinetics/families',
    families=test_fams)
total_nodes = 0
for db, src in [(real_db, 'real'), (test_db, 'test')]:
    for f in db.families:
        fam = db.families[f]
        g = fam.groups
        print(src, f, "top:", [e.label for e in g.top],
              "n_entries:", len(g.entries),
              "fwd_template:", [e.label for e in fam.forward_template.reactants],
              "rev_template:",
              ([e.label for e in fam.reverse_template.reactants]
               if fam.reverse_template is not None else None))
        total_nodes += len(g.entries)
print("TOTAL group-tree entries to record:", total_nodes)
