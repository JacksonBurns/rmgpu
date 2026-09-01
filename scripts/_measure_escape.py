import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')

name = 'H_Abstraction'
p = ('/home/jackson/rmgpu/RMG-database/input/kinetics/families/%s/groups.py'
     % name)
src = open(p).read()

captured_entries = []
captured_tree = None

def entry(*, index=None, label='', group='', **kw):
    captured_entries.append(label)

def tree(s):
    global captured_tree
    captured_tree = s

env = {'entry': entry, 'tree': tree, 'True': True, 'False': False,
       'None': None, 'template': lambda **k: None,
       'recipe': lambda **k: None, 'forbidden': lambda **k: None}
g = dict(env)
exec(compile(src, p, 'exec'), g)

# the entry label that has H2 and Cs|Cs
entry_hits = [e for e in captured_entries if 'H2' in e and 'Cs|Cs' in e]
print('entry labels with H2+Cs|Cs:')
for e in entry_hits:
    print('   ', repr(e), 'len', len(e))
print()
# tree lines with the same
print('tree lines with H2+Cs|Cs:')
for ln in captured_tree.splitlines():
    if 'H2' in ln and 'Cs|Cs' in ln and ln.strip().startswith('L'):
        lab = ln.strip().split(':', 1)[1].strip()
        print('   ', repr(lab), 'len', len(lab))
print()
# Compare: does any tree label equal any entry label (exact)?
tree_labels = set()
for ln in captured_tree.splitlines():
    if ln.strip().startswith('L'):
        lab = ln.strip().split(':', 1)[1].strip()
        tree_labels.add(lab)
en = set(captured_entries)
print('tree labels not in entries:', len(tree_labels - en))
for t in sorted(tree_labels - en):
    print('   TREE only:', repr(t))
    # nearest entry
    for e in sorted(en):
        if e.rstrip('|#O').rstrip() == t.rstrip('|#O').rstrip() or t in e or e in t:
            print('      candidate entry:', repr(e))
