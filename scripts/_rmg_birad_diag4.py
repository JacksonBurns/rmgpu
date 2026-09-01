"""Direct: does RMG-Py's same_species_lists say HOOH == HOOH? And re-run
get_labeled_reactants_and_products with full trace. Run in rmg_env."""
import json
import itertools
from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.molecule import Molecule
from rmgpy.reaction import Reaction, same_species_lists

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

reactants0 = [m.copy(deep=True) for m in oh]
template = fam.forward_template
grps = template.reactants[0].item.split()
template_reactants = list(grps)
mappings_a = fam._match_reactant_to_template(reactants0[0], template_reactants[0])
mappings_b = fam._match_reactant_to_template(reactants0[1], template_reactants[1])
print('mappings:', len(mappings_a), len(mappings_b))
for mapping in itertools.product(mappings_a, mappings_b):
    ps = fam._generate_product_structures(reactants0, mapping,
                                          forward=True, relabel_atoms=True)
    print('product_structures:', ps)
    if ps is None:
        print('  -> None (no product)')
        continue
    # compare each product to the target
    for p in ps:
        print('  product adjlist:\n   ', p.to_adjacency_list().replace('\n', '\n   '))
    # same_species_lists
    print('  same_species_lists(prod, ps) =',
          same_species_lists(list(prod), list(ps)))
    # individual isomorphism
    print('  prod[0] is_isomorphic p[0] =',
          prod[0].is_isomorphic(ps[0]))
    # with relabel_atoms=False
    ps2 = fam._generate_product_structures(reactants0, mapping,
                                           forward=True, relabel_atoms=False)
    if ps2 is not None:
        print('  relabel=False same_species_lists =',
              same_species_lists(list(prod), list(ps2)))
        print('  relabel=False p0 adjlist:\n   ',
              ps2[0].to_adjacency_list().replace('\n', '\n   '))
    # get_labeled
    out = fam.get_labeled_reactants_and_products(
        [m.copy(deep=True) for m in oh], [m.copy(deep=True) for m in prod])
    print('  get_labeled ->', out[0] is not None, out[1] is not None)
