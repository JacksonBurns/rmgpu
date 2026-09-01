p = ('/home/jackson/rmgpu/RMG-database/input/kinetics/families/'
     'H_Abstraction/groups.py')
lines = open(p).read().splitlines()
# print any line that contains a backslash (raw, as in file)
n = 0
for i, ln in enumerate(lines):
    if '\\' in ln:
        print('%4d %s' % (i + 1, ln))
        n += 1
        if n > 14:
            print('   ... (more)')
            break
print('total backslash lines in file:',
      sum(1 for ln in lines if '\\' in ln))
