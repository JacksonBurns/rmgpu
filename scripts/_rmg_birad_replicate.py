"""Replicate RMG-Py get_labeled_reactants_and_products EXACTLY for
Birad_recombination on OH+OH, calling the REAL Cython sub-methods, with
prints at every decision. Run in rmg_env."""
import json, itertools
from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.molecule import Molecule
from rmgpy.reaction import same_species_lists
from rmgpy.exceptions import ForbiddenStructureException, ActionError

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


def replicate(reactants, products, relabel_atoms=True):
    template = fam.forward_template
    reactants0 = [r.copy(deep=True) for r in reactants]
    print('auto_generated:', fam.auto_generated, 'reactant_num:',
          fam.reactant_num)
    if fam.auto_generated and fam.reactant_num != len(reactants):
        print('  -> rejected by auto_generated guard')
        return None, None
    if len(reactants) > len(template.reactants):
        grps = template.reactants[0].item.split()
        template_reactants = list(grps)
        print('split -> %d template_reactants' % len(template_reactants))
        for i, g in enumerate(template_reactants):
            print('   part%d atoms=%d labels=%s' % (
                i, len(g.atoms), [a.label for a in g.atoms]))
    else:
        template_reactants = [x.item for x in template.reactants]
    print('n reactants0=%d, n template_reactants=%d' % (
        len(reactants0), len(template_reactants)))
    molecule_a, molecule_b = reactants0
    ma_f = fam._match_reactant_to_template(molecule_a, template_reactants[0])
    mb_f = fam._match_reactant_to_template(molecule_b, template_reactants[1])
    print('forward: ma=%d mb=%d' % (len(ma_f), len(mb_f)))
    mappings = list(itertools.product(ma_f, mb_f))
    ma_r = fam._match_reactant_to_template(molecule_a, template_reactants[1])
    mb_r = fam._match_reactant_to_template(molecule_b, template_reactants[0])
    print('reverse: ma=%d mb=%d' % (len(ma_r), len(mb_r)))
    mappings.extend(list(itertools.product(ma_r, mb_r)))
    reactant_structures = [molecule_a, molecule_b]
    num_mappings = len(ma_f) * len(mb_f)
    print('total mappings to try:', len(mappings))
    for k, mapping in enumerate(mappings):
        try:
            ps = fam._generate_product_structures(
                reactant_structures, mapping, forward=True,
                relabel_atoms=relabel_atoms)
        except ForbiddenStructureException:
            print('  [%d] ForbiddenStructureException' % k)
            continue
        if ps is None:
            print('  [%d] product None' % k)
            continue
        ok = same_species_lists(list(products), list(ps),
                                save_order=fam.save_order)
        print('  [%d] product n=%d smiles=%s same_species=%s' % (
            k, len(ps), ps[0].to_smiles(), ok))
        if ok:
            print('  -> WOULD MATCH')
            return reactant_structures, ps
    print('num_mappings=%d' % num_mappings)
    if num_mappings > 0:
        print('  -> would raise ActionError (but ref says matched=False)')
    return None, None


print('save_order:', fam.save_order)
out = replicate(list(oh), list(prod))
print('REPLICATE RESULT:', out[0] is not None)
print('\n--- now the REAL method ---')
real = fam.get_labeled_reactants_and_products(
    [m.copy(deep=True) for m in oh], [m.copy(deep=True) for m in prod])
print('REAL RESULT:', real[0] is not None)
