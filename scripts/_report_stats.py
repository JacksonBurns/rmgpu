import sys, time
sys.path.insert(0, '.')
from rmgpu.core.family import KineticsFamilies
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import template as T
import json

t0 = time.time()
kf = KineticsFamilies().load('default')
t1 = time.time()
print('load: %.1fs, %d families, %d blocked' % (
    t1 - t0, len(kf.families), len(kf.blocked)))
print('rules loaded: %d/51, rules_error: %d' % (
    sum(1 for f in kf.families if f.rules is not None),
    sum(1 for f in kf.families if f.rules_error)))
print('forbidden present: %d' % sum(1 for f in kf.families if f.forbidden))

ref = json.load(open('gates/baselines/job05/step04_families_reference.json'))
t0 = time.time()
n = 0
for rec in ref['reactions']:
    rm = [Molecule.from_adjacency_list(a) for a in rec['reactants']]
    pm = [Molecule.from_adjacency_list(a) for a in rec['products']]
    f, l = kf.match_reaction(T.Reaction(rm, pm))
    if f is not None:
        n += 1
t1 = time.time()
print('match_reaction 20 reactions: matched %d, %.1fs' % (n, t1 - t0))
