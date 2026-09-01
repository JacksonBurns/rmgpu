import json

d = json.load(open('gates/baselines/job05/step04_families_reference.json'))
r = d['reactions'][0]
print('reaction keys:', list(r.keys()))
print('n reactants:', len(r['reactants']))
print('first reactant (trunc):', repr(r['reactants'][0][:120]))
print('n products:', len(r['products']))
f = d['families']['H_Abstraction']
print('H_Abstraction keys:', sorted(f.keys()))
print('top:', f['top'])
print('num_groups:', f['num_groups'], 'file:', f['num_groups_file'])
print('reactant_num:', f['reactant_num'], 'product_num:', f['product_num'])
print('eff tr:', f['num_template_reactants_effective'],
      'eff prod:', f['effective_product_num_forward'])
print('auto_gen:', f['auto_generated'], 'own_rev:', f['own_reverse'])
# a single-group family
rr = d['families']['R_Recombination']
print('R_Recomb tr:', rr['num_template_reactants'], 'eff:',
      rr['num_template_reactants_effective'])
# logic node sample
import collections
cnt = collections.Counter()
for name, fam in d['families'].items():
    for e in fam['entries']:
        if e.get('is_logic'):
            cnt[name] += 1
print('families with logic nodes:', dict(cnt))
