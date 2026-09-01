import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.core.family import _parse_groups_file, _load_tree

for name in ['H_Abstraction', 'R_Recombination']:
    p = '/home/jackson/rmgpu/RMG-database/input/kinetics/families/%s/groups.py' % name
    cap = _parse_groups_file(p)
    children, parent, top = _load_tree(cap['tree'])
    entry_labels = set(e['label'] for e in cap['entries'])
    tree_labels = set()
    for lab in top:
        tree_labels.add(lab)
    for par, kids in children.items():
        tree_labels.add(par)
        tree_labels.update(kids)
    missing = sorted(tree_labels - entry_labels)
    extra = sorted(entry_labels - tree_labels)
    print('=== %s ===' % name)
    print('  entry count:', len(entry_labels), 'tree labels:', len(tree_labels))
    print('  in-tree-no-entry (%d):' % len(missing))
    for m in missing[:12]:
        print('     ', repr(m))
    print('  entry-not-in-tree (%d):' % len(extra))
    for m in extra[:12]:
        print('     ', repr(m))
