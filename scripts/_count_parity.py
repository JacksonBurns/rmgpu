import sys, time, json
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.core.family import KineticsFamilies

REPO = '/home/jackson/rmgpu/rmgpu'
REF = json.load(open(REPO + '/gates/baselines/job05/step04_families_reference.json'))

t0 = time.time()
kf = KineticsFamilies()
kf.load('default')
dt = time.time() - t0
print('load time: %.1fs' % dt)
print('families loaded: %d, blocked: %d' % (len(kf.families), len(kf.blocked)))
print('reference default set size: %d' % len(REF['default']))

mism = []
for name in REF['default']:
    if name not in kf._families:
        mism.append((name, 'NOT LOADED', None, None))
        continue
    f = kf.get_family(name)
    rf = REF['families'][name]
    checks = {
        'num_groups_file': (f.num_groups_file, rf['num_groups_file']),
        'recipe_actions': (len(f.recipe.actions), len(rf['recipe'])),
        'own_reverse': (f.own_reverse, rf['own_reverse']),
        'reversible': (f.reversible, rf['reversible']),
        'auto_generated': (f.auto_generated, rf['auto_generated']),
        'allow_charged': (f.allow_charged_species, rf['allow_charged_species']),
        'electrons': (f.electrons, rf['electrons']),
        'reactant_num_eff': (f.reactant_num, rf['num_template_reactants_effective']),
        'product_num_fwd': (f.product_num_forward, rf['effective_product_num_forward']),
        'reverse_map': (f.reverse_map, rf['reverse_map']),
        'top': ([e.label for e in f.top], rf['top']),
        'fwd': ([e.label for e in f.forward_template], rf['template_reactants']),
        'template_products': (f.template_products, rf['template_products']),
    }
    for k, (got, want) in checks.items():
        if got != want:
            mism.append((name, k, got, want))
print('field mismatches:', len(mism))
for m in mism:
    print('   ', m)
# recipe content equality (actions)
rmm = []
for name in REF['default']:
    if name not in kf._families:
        continue
    f = kf.get_family(name)
    rf = REF['families'][name]
    got = [list(a) for a in f.recipe.actions]
    want = [list(a) for a in rf['recipe']]
    if got != want:
        rmm.append((name, got, want))
print('recipe content mismatches:', len(rmm))
for m in rmm[:6]:
    print('   ', m[0], 'got', m[1][:3], 'want', m[2][:3])
# reverse recipe presence
rv = []
for name in REF['default']:
    if name not in kf._families:
        continue
    f = kf.get_family(name)
    rf = REF['families'][name]
    has_rr = f.reverse_recipe is not None
    if rf['reverse_recipe'] is None and not has_rr:
        continue
    if rf['reverse_recipe'] is not None:
        got = [list(a) for a in f.reverse_recipe.actions] if f.reverse_recipe else None
        if got != [list(a) for a in rf['reverse_recipe']]:
            rv.append((name, got, rf['reverse_recipe']))
print('reverse_recipe mismatches:', len(rv))
for m in rv[:6]:
    print('   ', m[0])
print('blocked:', list(kf.blocked))
