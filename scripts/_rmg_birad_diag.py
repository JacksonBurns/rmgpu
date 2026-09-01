"""Diagnose why RMG-Py says Birad_recombination does NOT match OH+OH -> HOOH
(the step-04 reference records matched=False). Run in rmg_env."""
from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.molecule import Molecule
import os

FAM = settings['database.directory'] + '/kinetics/families'

def mol(adj):
    return Molecule().from_adjacency_list(adj)

# OH and OH -> HOOH (the step-04 reference reaction R_Recombination|14|0)
import json
ref = json.load(open('/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step04_families_reference.json'))
rec = ref['reactions'][0]  # find the right one
# locate by ci/ri
for r in ref['reactions']:
    if (r['ci'], r['ri']) == (14, 0):
        rec = r
        break
print('reaction:', rec['own'], rec['ci'], rec['ri'])
oh = [mol(a) for a in rec['reactants']]
prod = [mol(a) for a in rec['products']]
def _formula(m):
    try:
        return m.molecular_formula
    except Exception:
        try:
            return str(m.formula)
        except Exception:
            return '?'
print('reactants:', [_formula(m) for m in oh])
print('products :', [_formula(m) for m in prod])

db = KineticsDatabase()
db.load_families(path=FAM, families=['Birad_recombination', 'R_Recombination'])

for famname in ('Birad_recombination', 'R_Recombination'):
    fam = db.families[famname]
    print('\n=== %s ===' % famname)
    print('reactant_num (fam):', fam.reactant_num,
          'template.reactants:', [e.label for e in fam.forward_template.reactants])
    grp = fam.forward_template.reactants[0].item
    parts = grp.split()
    print('Root split ->', len(parts),
          [len(p.atoms) for p in parts],
          [[a.label for a in p.atoms] for p in parts])
    try:
        out = fam.get_labeled_reactants_and_products(list(oh), list(prod))
        print('get_labeled ->', out[0] is not None)
    except Exception as e:
        print('get_labeled EXC:', type(e).__name__, str(e)[:200])
    # probe the single reactant match
    try:
        maps0 = fam._match_reactant_to_template(oh[0], parts[0])
        print('match(oh0, part0):', len(maps0))
        maps1 = fam._match_reactant_to_template(oh[0], parts[1])
        print('match(oh0, part1):', len(maps1))
        maps2 = fam._match_reactant_to_template(oh[1], parts[1])
        print('match(oh1, part1):', len(maps2))
    except Exception as e:
        print('match probe EXC:', type(e).__name__, str(e)[:200])
