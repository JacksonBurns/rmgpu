#!/usr/bin/env python3
"""Check: rmgpu replay raw-reaction count/order vs recorded n_raw_reactions /
raw_templates length, for all 30 cases. If they match per case, the replay
drops the same applications as RMG and raw_templates can be zipped by index."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import enumeration as enum

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'gates', 'baselines', 'job05', 'step02_products_reference.json')


def main():
    data = json.load(open(BASE))
    all_ok = True
    for case in data['cases']:
        key = case['family'] + '|' + ','.join(case['reactant_smiles'])
        fam = enum.Family.from_reference(case)
        match = enum.RecordedMatcher(case)
        rxn_list = []
        exc = None
        try:
            for structures in match.yield_applications():
                enum._apply_one_application(fam, structures, True, rxn_list)
        except Exception as e:
            exc = '%s: %s' % (type(e).__name__, e)
        n_raw_got = len(rxn_list)
        n_raw_want = case['n_raw_reactions']
        n_tmpl = len(case.get('raw_templates', []))
        status = 'OK'
        if exc:
            status = 'EXC %s' % exc
            all_ok = False
        elif n_raw_got != n_raw_want or n_raw_want != n_tmpl:
            status = 'MISMATCH got=%d want=%d tmpl=%d' % (n_raw_got, n_raw_want, n_tmpl)
            all_ok = False
        print('%-46s %s' % (key, status))
    print('ALL OK' if all_ok else 'SOME MISMATCH')


if __name__ == '__main__':
    main()
