p = ('/home/jackson/rmgpu/RMG-database/input/kinetics/families/'
     'H_Abstraction/groups.py')
raw = open(p, 'rb').read().decode('utf-8')
lines = raw.splitlines()
# entry label lines containing 'Cs|Cs' with a backslash somewhere
print('=== entry lines with backslash ending Cs|Cs ===')
for i, ln in enumerate(lines):
    if 'label=' in ln and '\\' in ln and 'Cs|Cs' in ln:
        print('%5d %s' % (i + 1, ln.strip()))
print()
print('=== tree lines containing \\\\H2\\\\Cs|Cs (double-backslash) ===')
for i, ln in enumerate(lines):
    if 'H2' in ln and 'Cs|Cs' in ln and ln.strip().startswith('L'):
        nb = ln.count('\\')
        print('%5d (nb=%d) %s' % (i + 1, nb, ln.strip()))
