"""Instrument RMG-Py's Birad_recombination matching for OH+OH -> HOOH to
find the exact rejection point. Run in rmg_env."""
import json
from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.molecule import Molecule

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

db = KineticsDatabase()
db.load_families(path=FAM, families=['Birad_recombination'])
fam = db.families['Birad_recombination']

template = fam.forward_template
reactants0 = [m.copy(deep=True) for m in oh]
grps = template.reactants[0].item.split()
template_reactants = list(grps)
print('n template_reactants after split:', len(template_reactants))
for i, g in enumerate(template_reactants):
    print('  part%d: atoms=%d labels=%s' % (
        i, len(g.atoms), [a.label for a in g.atoms]))

molecule_a, molecule_b = reactants0
mappings_a = fam._match_reactant_to_template(molecule_a, template_reactants[0])
mappings_b = fam._match_reactant_to_template(molecule_b, template_reactants[1])
print('mappings_a (oh0->part0):', mappings_a)
print('mappings_b (oh1->part1):', mappings_b)
import itertools
mappings = list(itertools.product(mappings_a, mappings_b))
print('total mappings:', len(mappings))

for mapping in mappings:
    print('\n--- mapping:', mapping)
    try:
        ps = fam._generate_product_structures(reactants0, mapping,
                                              forward=True,
                                              relabel_atoms=True)
        print('  product_structures:', None if ps is None else ps)
        if ps is not None:
            from rmgpy.molecule import same_species_lists
            ok = same_species_lists(list(prod), list(ps),
                                    save_order=fam.save_order)
            print('  same_species_lists vs target:', ok)
            for p in ps:
                print('    product adjlist:\n', p.to_adjacency_list())
    except Exception as e:
        print('  _generate_product_structures EXC:', type(e).__name__,
              str(e)[:300])
