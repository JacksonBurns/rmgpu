import json

d = json.load(open('gates/baselines/job05/step04_families_reference.json'))
v = d['verdicts']
for key in sorted(v):
    row = v[key]
    ms = [(n, val['template']) for n, val in row.items() if val['matched']]
    own = key.split('|')[0]
    mark = 'OK ' if own in [m[0] for m in ms] else 'XX '
    print(mark, key, '->', ms)
