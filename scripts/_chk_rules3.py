import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.core.family import KineticsFamilies
from pathlib import Path
import importlib

kf = KineticsFamilies().load('default')
fails = [f.name for f in kf.families if f.rules_error]

# Replicate _parse_rules_file's exec but capture the exception text
def parse_capture(path):
    from rmgpu.core.family import _parse_rules_file
    # monkeypatch: re-implement the try/except with a traceback
    import rmgpu.core.family as F
    import types
    path = Path(path)
    rules = []

    class _KineticsStub:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    names = ['Arrhenius', 'ArrheniusEP', 'ArrheniusBM', 'PDepArrhenius',
             'MultiPDepArrhenius', 'Lindemann', 'Troe', 'ThirdBody',
             'MultiArrhenius', 'Chebyshev', 'Marcus', 'KineticsData',
             'SurfaceArrhenius', 'SurfaceArrheniusBEP',
             'SurfaceChargeTransfer', 'ArrheniusChargeTransfer',
             'ArrheniusChargeTransferBM', 'StickingCoefficient',
             'StickingCoefficientBEP']
    stubs = {n: type(n, (_KineticsStub,), {}) for n in names}

    def entry(*, index=None, label='', kinetics=None, shortDesc='',
              longDesc='', rank=None, allow_max_rate_violation=None,
              reversible=None, elementary_high_p=None, duplicate=None,
              **kw):
        rules.append(label)

    env = {'entry': entry}
    env.update(stubs)
    env['R'] = 8.314472
    env['True'] = True
    env['False'] = False
    env['None'] = None
    import traceback
    try:
        g = dict(env)
        with open(path) as f:
            src = f.read()
        exec(compile(src, str(path), 'exec'), g)
        return ('OK', len(rules))
    except Exception:
        return ('ERR', traceback.format_exc().splitlines()[-3:])

# show the failing line for each distinct error
seen = {}
for n in fails:
    rp = Path('/home/jackson/rmgpu/RMG-database/input/kinetics/families') / n / 'rules.py'
    if not rp.exists():
        print('%s: NO rules.py' % n)
        continue
    st, info = parse_capture(rp)
    key = tuple(info) if st == 'ERR' else 'OK'
    if key not in seen:
        seen[key] = []
    seen[key].append(n)
for key, names in seen.items():
    print('\n=== %d families ===' % len(names))
    if key != 'OK':
        print('  error tail:', key)
    print('  ', names[:6])
