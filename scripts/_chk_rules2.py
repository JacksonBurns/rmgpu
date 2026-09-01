import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.core.family import KineticsFamilies

kf = KineticsFamilies().load('default')
fails = [f.name for f in kf.families if f.rules_error]
print('rules_error families (%d):' % len(fails))
for n in fails:
    print('  ', n)

# Now find the actual exception for one of them by re-running the parse
from rmgpu.core.family import _parse_rules_file
from pathlib import Path
import traceback
for n in fails[:3]:
    rp = Path('/home/jackson/rmgpu/RMG-database/input/kinetics/families') / n / 'rules.py'
    if not rp.exists():
        print('\n%s: no rules.py' % n)
        continue
    # replicate the try/except but capture the exception
    import ast
    src = rp.read_text()
    print('\n=== %s (first 400 chars) ===' % n)
    print(src[:400])
