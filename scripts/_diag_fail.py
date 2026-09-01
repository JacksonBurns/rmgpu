import sys, traceback
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.core.family import Family, DEFAULT_FAMILIES_DIR, DEFAULT_KINETICS_DB

for name in ['H_Abstraction', 'R_Recombination']:
    try:
        fam = Family.from_files(name, DEFAULT_FAMILIES_DIR, DEFAULT_KINETICS_DB)
        print(name, 'OK: top', [e.label for e in fam.top][:5],
              'fwd', [e.label for e in fam.forward_template],
              'rn', fam.reactant_num)
    except Exception as e:
        print(name, 'FAIL:', type(e).__name__, str(e)[:500])
