import sys, time, traceback
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.core.family import KineticsFamilies, Family

t0 = time.time()
kf = KineticsFamilies()
kf.load('default')
t1 = time.time()
print('loaded: %d families, %d blocked in %.1fs' % (
    len(kf.families), len(kf.blocked), t1 - t0))
if kf.blocked:
    for name, info in kf.blocked.items():
        print('  BLOCKED:', name, '->', info['reason'][:300])

# compare counts with the recorded reference
import json
ref = json.load(open(
    '/home/jackson/rmgpu/rmgpu/gates/baselines/job05/'
    'step04_families_reference.json'))
mism = []
for name in ref['default']:
    f = kf.get_family(name)
    rf = ref['families'][name]
    checks = {
        'reactant_num_eff': (f.reactant_num,
                             rf['num_template_reactants_effective']),
        'product_num_fwd': (f.product_num_forward,
                            rf['effective_product_num_forward']),
        'num_groups': (f.num_groups_file, rf['num_groups_file']),
        'own_reverse': (f.own_reverse, rf['own_reverse']),
        'reversible': (f.reversible, rf['reversible']),
        'auto_generated': (f.auto_generated, rf['auto_generated']),
        'top': (f.top_labels if hasattr(f, 'top_labels') else
                [e.label for e in f.top], rf['top']),
        'template_reactants': ([e.label for e in f.forward_template],
                               rf['template_reactants']),
    }
    for k, (got, want) in checks.items():
        if got != want:
            mism.append((name, k, got, want))
print('count mismatches vs reference:', len(mism))
for m in mism[:40]:
    print('  ', m)
