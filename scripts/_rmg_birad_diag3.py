"""Compare RMG's generated product for R_Recombination vs Birad_recombination
on OH+OH -> HOOH. Run in rmg_env."""
import json
from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.molecule import Molecule
import itertools

FAM = settings['database.directory'] + '/kinetics/families'


def mol(adj):
    return Molecule().from_adjacency_list(adj)


ref = json.load(open(
    '/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step04_families_reference.json'))
for r in ref['reactions']:
    if (r['ci'], r['ri']) == (14, 0):
        rec = r
        break
oh = [mol(a) for a in rec['reactants']]
prod = [mol(a) for a in rec['products']]
print('TARGET product adjlist:\n', prod[0].to_adjacency_list())

db = KineticsDatabase()
db.load_families(path=FAM, families=['Birad_recombination', 'R_Recombination'])

for famname in ('R_Recombination', 'Birad_recombination'):
    fam = db.families[famname]
    print('\n===== %s =====' % famname)
    print('recipe:', [(a.__class__.__name__, getattr(a, 'atom1', None),
                      getattr(a, 'atom2', None), getattr(a, 'bondOrder', None))
                     for a in fam.forward_recipe.actions])
    reactants0 = [m.copy(deep=True) for m in oh]
    template = fam.forward_template
    if len(reactants0) > len(template.reactants):
        grps = template.reactants[0].item.split()
        template_reactants = list(grps)
    else:
        template_reactants = [x.item for x in template.reactants]
    ma = fam._match_reactant_to_template(reactants0[0], template_reactants[0])
    mb = fam._match_reactant_to_template(reactants0[1], template_reactants[1])
    for mapping in itertools.product(ma, mb):
        try:
            ps = fam._generate_product_structures(reactants0, mapping,
                                                  forward=True,
                                                  relabel_atoms=True)
        except Exception as e:
            print('  EXC:', type(e).__name__, str(e)[:200])
            continue
        if ps is None:
            print('  product_structures None')
            continue
        for p in ps:
            print('  PRODUCT:\n', p.to_adjacency_list())
