import json

p = '/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step03_templates_reference.json'
d = json.load(open(p))
print('top-level keys:', list(d.keys()))
fam = d['families']
print('n families:', len(fam))
for name in ['1,2_shiftC', 'H_Abstraction', 'R_Recombination', 'intra_H_migration']:
    if name not in fam:
        print(name, 'NOT IN step03')
        continue
    f = fam[name]
    print('---', name, '---')
    print('  keys:', sorted(f.keys()))
    print('  top:', f.get('top'))
    print('  forward_template:', f.get('forward_template'))
    tree = f.get('tree', {})
    # count root nodes in tree (nodes with parent None)
    roots = [lab for lab, node in tree.items() if not node.get('parent')]
    print('  tree roots (no parent):', sorted(roots))
