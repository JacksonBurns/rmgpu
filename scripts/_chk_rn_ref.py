import json

d = json.load(open('gates/baselines/job05/step04_families_reference.json'))
print('family          reactant_num(stored)  eff  product_num')
for name in ('R_Recombination', 'Birad_recombination',
             'R_Addition_MultipleBond', 'H_Abstraction',
             'intra_H_migration', '1,2_shiftC'):
    f = d['families'][name]
    print('%-24s  %-22s %s   %s' % (
        name, f['reactant_num'],
        f['num_template_reactants_effective'], f['product_num']))
# how many families have reactant_num None (use slot count)
n_none = sum(1 for f in d['families'].values() if f['reactant_num'] is None)
print('families with reactant_num None (slot count):', n_none,
      'of', len(d['families']))
