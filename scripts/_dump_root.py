import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.core.family import _parse_groups_file
import json

fam = _parse_groups_file(
    '/home/jackson/rmgpu/RMG-database/input/kinetics/families/'
    'R_Recombination/groups.py')
for e in fam['entries']:
    if e['label'] in ('Root', 'Y_rad', 'Root_1R->H'):
        print('=== entry %s ===' % e['label'])
        print(e['group'][:600])
        print()
