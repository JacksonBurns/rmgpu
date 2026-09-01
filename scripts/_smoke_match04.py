import sys, time, json
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.core.family import KineticsFamilies
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import template as T

REPO = '/home/jackson/rmgpu/rmgpu'
REF = json.load(open(REPO + '/gates/baselines/job05/step04_families_reference.json'))

kf = KineticsFamilies()
t0 = time.time()
kf.load('default')
t1 = time.time()
print('loaded %d families, %d blocked in %.1fs' % (
    len(kf.families), len(kf.blocked), t1 - t0))
for name, info in kf.blocked.items():
    print('  BLOCKED:', name, '->', info['reason'][:200])

# top / forward_template comparison vs reference for mechanism families
print('\n--- top vs reference ---')
mech = ['H_Abstraction', 'R_Recombination', 'R_Addition_MultipleBond',
        'intra_H_migration', 'Intra_ene_reaction', '1,2_shiftC',
        'Singlet_Val6_to_triplet']
for name in mech:
    if name not in kf._families:
        print('%s: NOT LOADED' % name)
        continue
    f = kf.get_family(name)
    rf = REF['families'][name]
    got_top = [e.label for e in f.top]
    got_fwd = [e.label for e in f.forward_template]
    print('%s:' % name)
    print('   my top:', got_top)
    print('   ref top:', rf['top'])
    print('   my fwd:', got_fwd, ' ref template_reactants:',
          rf['template_reactants'])
    print('   match top==ref:', got_top == rf['top'],
          ' fwd==reactants:', got_fwd == rf['template_reactants'])

# now the match_reaction checks
print('\n--- match_reaction (20 cases) ---')
good = 0
bad = []
for rec in REF['reactions']:
    key = '%s|%d|%d' % (rec['own'], rec['ci'], rec['ri'])
    reactants = [Molecule.from_adjacency_list(a) for a in rec['reactants']]
    products = [Molecule.from_adjacency_list(a) for a in rec['products']]
    rxn = T.Reaction(reactants, products)
    fam, labels = kf.match_reaction(rxn)
    got_name = fam.label if fam is not None else None
    # reference: which family matched?
    row = REF['verdicts'][key]
    ref_matchers = [n for n, v in row.items() if v['matched']]
    ref_labels = None
    if len(ref_matchers) == 1:
        ref_labels = row[ref_matchers[0]]['template']
    ok = (got_name in ref_matchers)
    label_ok = (labels == ref_labels) if (ok and ref_labels) else None
    if ok:
        good += 1
    else:
        bad.append((key, got_name, ref_matchers, labels, ref_labels))
    print('  %-52s got=%s ref=%s labels_ok=%s' % (
        key, got_name, ref_matchers, label_ok))
print('\nmatch_reaction: %d/20 correct family, %d bad' % (good, len(bad)))
for b in bad:
    print('   BAD:', b)
