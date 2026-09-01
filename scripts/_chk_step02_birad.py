import json

p = 'gates/baselines/job05/step02_products_reference.json'
d = json.load(open(p))
print('step-02 reference: cases =', len(d['cases']))
for i, c in enumerate(d['cases']):
    fam = c['family']
    if 'Birad' in fam or fam == 'R_Recombination':
        print('\ncase %d: family=%s' % (i, fam))
        meta = c.get('family_meta', {})
        print('  meta reactants_eff:', meta.get('num_template_reactants_effective'),
              'template:', meta.get('template_reactants') or
              meta.get('template'))
        for j, rxn in enumerate(c['reactions'][:3]):
            print('  rxn %d: %d reactants %d products' % (
                j, len(rxn['reactants']), len(rxn['products'])))
            print('    reactants:', [r.splitlines()[0] if r else '' for r in
                                     rxn['reactants']])
            print('    products :', [r.splitlines()[0] if r else '' for r in
                                     rxn['products']])
