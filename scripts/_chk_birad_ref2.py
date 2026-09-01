import json
d = json.load(open('gates/baselines/job05/step04_families_reference.json'))
for k in ('R_Recombination|14|0', 'R_Recombination|15|0'):
    v = d['verdicts'][k]['Birad_recombination']
    print(k, 'Birad_recombination:', v)
    print(k, 'R_Recombination:   ', d['verdicts'][k]['R_Recombination'])
