"""Instrument the REAL get_labeled_reactants_and_products for
Birad_recombination on OH+OH by monkeypatching the two helpers. Run in rmg_env."""
import json, itertools
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

orig_gen = fam.__class__._generate_product_structures
orig_match = fam.__class__._match_reactant_to_template


def trace_gen(self, reactants, mapping, **kw):
    try:
        ps = orig_gen(self, reactants, mapping, **kw)
    except Exception as e:
        print('  _gen EXC:', type(e).__name__, str(e)[:150])
        raise
    s = 'None' if ps is None else 'n=%d' % len(ps)
    print('  _gen ->', s,
          (ps[0].to_smiles() if ps else ''))
    return ps


def trace_match(self, m, group):
    r = orig_match(self, m, group)
    print('  _match(%s) -> %d mappings' % (
        m.get_formula() if hasattr(m, 'get_formula') else m, len(r)))
    return r


fam.__class__._generate_product_structures = trace_gen
fam.__class__._match_reactant_to_template = trace_match

print('calling real get_labeled_reactants_and_products:')
out = fam.get_labeled_reactants_and_products(
    [m.copy(deep=True) for m in oh], [m.copy(deep=True) for m in prod])
print('RESULT:', out[0] is not None, out[1] is not None)
