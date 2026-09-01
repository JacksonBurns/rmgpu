import json

d = json.load(open('gates/baselines/job05/step04_families_reference.json'))
mis = []
for name, f in d['families'].items():
    tr = f['template_reactants']
    top = f['top']
    if top != tr:
        mis.append((name, tr, top))
print('families where top != template_reactants:', len(mis))
for m in mis:
    print('  ', m[0])
    print('    template_reactants:', m[1])
    print('    top:', m[2][:6], '...' if len(m[2]) > 6 else '')
# also: how many template products (families with non-empty template products)
nprod = {n: f['template_products'] for n, f in d['families'].items()
         if f['template_products']}
print('families with template products:', len(nprod))
for n, p in nprod.items():
    print('   ', n, p)
