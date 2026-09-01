import sys, time, json
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.core.family import KineticsFamilies
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import template as T

REPO = '/home/jackson/rmgpu/rmgpu'
REF = json.load(open(REPO + '/gates/baselines/job05/step04_families_reference.json'))
kf = KineticsFamilies()
kf.load('default')

t0 = time.time()
for rec in REF['reactions']:
    key = '%s|%d|%d' % (rec['own'], rec['ci'], rec['ri'])
    reactants = [Molecule.from_adjacency_list(a) for a in rec['reactants']]
    products = [Molecule.from_adjacency_list(a) for a in rec['products']]
    rxn = T.Reaction(reactants, products)
    fam, labels = kf.match_reaction(rxn)
    if key in ('H_Abstraction|4|0', 'R_Addition_MultipleBond|19|0'):
        row = REF['verdicts'][key]
        refm = [n for n, v in row.items() if v['matched']]
        print(key)
        print('   my labels:   ', labels)
        print('   ref labels:  ', row[refm[0]]['template'])
print('total match_reaction time for 20 reactions: %.1fs' % (time.time() - t0))
