import sys, json
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.core.family import KineticsFamilies
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import template as T

REPO = '/home/jackson/rmgpu/rmgpu'
REF = json.load(open(REPO + '/gates/baselines/job05/step04_families_reference.json'))
kf = KineticsFamilies().load('default')

for r in REF['reactions']:
    if (r['ci'], r['ri']) == (14, 0):
        rec = r
        break
rm = [Molecule.from_adjacency_list(a) for a in rec['reactants']]
pm = [Molecule.from_adjacency_list(a) for a in rec['products']]
rxn = T.Reaction(rm, pm)
print('reactants:', [m.get_formula() for m in rm])

# what does each of the 3 recombination families match?
for name in ('R_Recombination', 'Birad_recombination',
             'Birad_R_Recombination', 'Disproportionation'):
    fam = kf.get_family(name)
    print('\n=== %s ===' % name)
    print('  reactant_num:', fam.reactant_num,
          'fwd:', [e.label for e in fam.forward_template],
          'top:', [e.label for e in fam.top])
    got = T.match(fam, rxn)
    print('  T.match ->', got)
