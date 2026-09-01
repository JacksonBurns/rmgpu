import re

p = ('/home/jackson/rmgpu/RMG-database/input/kinetics/families/'
     'H_Abstraction/groups.py')
src = open(p).read()

# find the entry with a backslash label
for m in re.finditer(r"entry\(\s*index=\d+,\s*label=('[^']*'|\"[^\"]*\"),", src):
    lab = m.group(1)
    if '\\' in lab:
        print('ENTRY label (raw):', repr(lab))
        # unescape like python
        print('  unescaped:', repr(eval(lab)))

print()
# find tree lines with backslash
in_tree = False
buf = []
for ln in src.splitlines():
    if ln.startswith('tree('):
        in_tree = True
        continue
    if in_tree:
        if ln.strip() == ')':
            break
        buf.append(ln)
tree = '\n'.join(buf)
# The captured tree string, after Python unescape of the triple-quote:
import ast
# reconstruct: find the triple-quoted block
tm = re.search(r'tree\(\s*(r?"""|r?\'\'\')(.*?)\1\s*\)', src, re.S)
print('tree capture group0:', repr(tm.group(1)) if tm else 'NONE')
raw_block = tm.group(2) if tm else ''
unesc = eval('"""%s"""' % raw_block)
# find tree lines with backslash
for tln in unesc.splitlines():
    if '\\' in tln:
        print('TREE line (unescaped):', repr(tln.strip()))
