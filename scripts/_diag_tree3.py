import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.core.family import _parse_groups_file, _load_tree

name = 'H_Abstraction'
p = '/home/jackson/rmgpu/RMG-database/input/kinetics/families/%s/groups.py' % name
cap = _parse_groups_file(p)
children, parent, top = _load_tree(cap['tree'])
entry_labels = set(e['label'] for e in cap['entries'])
tree_labels = set(top)
for par, kids in children.items():
    tree_labels.add(par)
    tree_labels.update(kids)
missing = sorted(tree_labels - entry_labels)
print('missing (in tree, no entry):', len(missing))
# For a couple, find the closest entry label (by ignoring backslashes)
for m in missing[:6]:
    cand = [e for e in entry_labels if e.replace('\\','')==m.replace('\\','')]
    print('  tree: %r' % m)
    for c in cand[:3]:
        print('    entry-candidate: %r' % c)
# Count how many tree labels have backslash
print('tree labels with backslash:', sum(1 for t in tree_labels if '\\' in t))
print('entry labels with backslash:', sum(1 for t in entry_labels if '\\' in t))
# Show one tree label and its nearest entry with full repr
m = missing[0]
print()
print('MISMATCHED LABEL full repr:')
print('  tree :', repr(m))
cand = [e for e in entry_labels if e.replace('\\','')==m.replace('\\','')]
if cand:
    print('  entry:', repr(cand[0]))
    # show the diff char by char
    a, b = m, cand[0]
    for i in range(max(len(a), len(b))):
        ca = a[i] if i < len(a) else '.'
        cb = b[i] if i < len(b) else '.'
        if ca != cb:
            print('   first diff at %d: tree=%r entry=%r' % (i, ca, cb))
            break
