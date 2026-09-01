import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.core.family import KineticsFamilies

kf = KineticsFamilies().load('default')
n_rules = 0
n_rules_err = 0
n_forbidden = 0
examples = {}
for f in kf.families:
    if f.rules is not None:
        n_rules += 1
        examples[f.name] = len(f.rules)
    if f.rules_error:
        n_rules_err += 1
    if f.forbidden:
        n_forbidden += 1
print('families with rules (DATA):', n_rules, 'of', len(kf.families))
print('families with rules_error:', n_rules_err)
print('families with forbidden structures:', n_forbidden)
# show one rules example
for name in ('H_Abstraction', 'R_Recombination', 'intra_H_migration'):
    f = kf.get_family(name)
    print('\n%s: %d rules' % (name, len(f.rules or [])))
    for r in (f.rules or [])[:2]:
        print('   ', r['label'], 'kind=%s rank=%s' % (r['kinetics_kind'], r['rank']))
# confirm rules args are raw (data), not evaluated
f = kf.get_family('H_Abstraction')
r0 = f.rules[0]
print('\nrule[0] kinetics_args type:', type(r0['kinetics_args']))
print('  A:', r0['kinetics_args'][0][:1] if r0['kinetics_args'][0] else None)
