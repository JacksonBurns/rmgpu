p = ('/home/jackson/rmgpu/RMG-database/input/kinetics/families/'
     'H_Abstraction/groups.py')
raw = open(p).read()
lines = raw.splitlines()

# The unescaped target label: C_rad/H2/Cs\H2\Cs|Cs  (single backslashes)
target = 'C_rad/H2/Cs\\H2\\Cs|Cs'   # python literal -> single backslashes

print('target (single backslashes):', repr(target))
print()
# entry lines: find entry whose label string, after python unescape, == target
import ast
found_entry = False
for i, ln in enumerate(lines):
    s = ln.strip()
    if s.startswith('entry(') and 'label=' in s:
        # extract label='...' or label="..."
        import re
        m = re.search(r"label=(['\"])(.*?)\1", s)
        if m:
            lit = m.group(1) + m.group(2) + m.group(1)
            try:
                lab = ast.literal_eval(lit)
            except Exception:
                continue
            if lab == target:
                print('ENTRY match line %d (raw):' % (i + 1))
                print('   ', s[:200])
                found_entry = True
            elif target in lab or lab in target:
                print('NEAR entry line %d: %r' % (i + 1, lab))
print('entry found:', found_entry)
print()
# tree lines: find whose raw label == target (after the L<N>: prefix)
for i, ln in enumerate(lines):
    s = ln.strip()
    m = re.match(r'^L\d+:\s*(\S+)$', s)
    if m:
        # the raw label text (before python unescape)
        rl = m.group(1)
        if 'H2' in rl and 'Cs|Cs' in rl and 'O' not in rl:
            print('TREE raw line %d: %s' % (i + 1, s))
            print('   raw label repr:', repr(rl))
