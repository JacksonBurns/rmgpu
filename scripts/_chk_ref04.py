import json

d = json.load(open('gates/baselines/job05/step04_families_reference.json'))
print('keys:', list(d.keys()))
print('default set size:', len(d['default']))
print('families:', len(d['families']))
print('match cases:', len(d['match_cases']))
print('reactions:', len(d['reactions']))
print('verdict keys:', len(d['verdicts']))
tot = sum(f['num_groups'] for f in d['families'].values())
totf = sum(f['num_groups_file'] for f in d['families'].values())
print('total group entries loaded:', tot, 'file entries:', totf)
# how many reactions have >=1 matching family
multi = 0
for k, row in d['verdicts'].items():
    ms = [n for n, v in row.items() if v['matched']]
    if len(ms) > 1:
        multi += 1
        print('multi-match:', k, ms)
print('multi-match reactions:', multi)
