import json

d = json.load(open('gates/baselines/job05/step04_families_reference.json'))
# R_Recombination|14|0 = OH + OH -> HOOH ; 15|0 = CH3 + OH
for key in ('R_Recombination|14|0', 'R_Recombination|15|0'):
    row = d['verdicts'][key]
    print('=== %s ===' % key)
    for fam in ('R_Recombination', 'Birad_recombination',
                'Birad_R_Recombination', 'Disproportionation'):
        v = row.get(fam)
        if v is None:
            print('  %-22s NOT IN verdicts' % fam)
        else:
            print('  %-22s matched=%s template=%s error=%s' % (
                fam, v['matched'], v['template'],
                (v['error'] or '')[:60]))
    # who matched
    ms = [n for n, v in row.items() if v['matched']]
    print('  ALL matchers:', ms)
# reference effective reactant counts
for fam in ('R_Recombination', 'Birad_recombination', 'Birad_R_Recombination'):
    f = d['families'][fam]
    print('--- %s ---' % fam)
    print('  reactant_num(stored):', f['reactant_num'],
          'eff:', f['num_template_reactants_effective'])
    print('  template_reactants:', f['template_reactants'])
    print('  own_reverse:', f['own_reverse'], 'reversible:', f['reversible'])
    print('  recipe:', f['recipe'])
