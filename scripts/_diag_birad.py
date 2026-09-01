import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.core.family import _parse_groups_file
from rmgpu.molecule.group import Group

p = ('/home/jackson/rmgpu/RMG-database/input/kinetics/families/'
     'Birad_recombination/groups.py')
cap = _parse_groups_file(p)
for e in cap['entries']:
    if e['label'] == 'Root':
        print('Root group:')
        print(e['group'])
        g = Group().parse(e['group'])
        print('n atoms:', len(g.atoms))
        parts = g.split()
        print('my split() ->', len(parts), 'components')
        for i, pg in enumerate(parts):
            print('   comp %d: %d atoms, %d bonds, labels=%s' % (
                i, len(pg.atoms), len(pg.edges),
                [a.label for a in pg.atoms]))
        # labels
        print('all labels:', [a.label for a in g.atoms])
