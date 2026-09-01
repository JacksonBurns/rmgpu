#!/usr/bin/env python3
"""Probe: dump per-atom / per-bond detail of rmgpu product pieces for one
case, to see why _mols_identical fails on equal atom-ID sets."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rmgpu.molecule.molecule import Molecule
from rmgpu.core import enumeration as enum

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'gates', 'baselines', 'job05', 'step02_products_reference.json')


def dump_piece(name, m):
    print('  piece %s:' % name)
    for a in m._rdkit.GetAtoms():
        aid = a.GetProp('atomid') if a.HasProp('atomid') else '?'
        bds = []
        for b in a.GetBonds():
            o = b.GetBeginAtomIdx() if b.GetBeginAtomIdx() != a.GetIdx() else b.GetEndAtomIdx()
            oa = m._rdkit.GetAtomWithIdx(o)
            oaid = oa.GetProp('atomid') if oa.HasProp('atomid') else '?'
            bds.append((oaid, b.GetBondTypeAsDouble(), b.GetBondType()))
        print('    id=%s %s rad=%d chg=%d lp=%s bonds=%s' % (
            aid, a.GetSymbol(), a.GetNumRadicalElectrons(), a.GetFormalCharge(),
            a.GetProp('lp') if a.HasProp('lp') else '-', sorted(bds)))


def main():
    key = sys.argv[1]
    i0, i1 = int(sys.argv[2]), int(sys.argv[3])
    data = json.load(open(BASE))
    for case in data['cases']:
        if case['family'] + '|' + ','.join(case['reactant_smiles']) != key:
            continue
        fam = enum.Family.from_reference(case)
        match = enum.RecordedMatcher(case)
        rxns = []
        for structures in match.yield_applications():
            enum._apply_one_application(fam, structures, True, rxns)
        r0, r1 = rxns[i0], rxns[i1]
        print(key, 'rxn%d vs rxn%d' % (i0, i1))
        for k in range(len(r0.products)):
            print(' product %d:' % k)
            dump_piece('  rxn%d' % i0, r0.products[k])
            dump_piece('  rxn%d' % i1, r1.products[k])
        print('  identical_species_lists:',
              enum.identical_species_lists(r0.products, r1.products))
        for k in range(len(r0.products)):
            print('  _mols_identical(piece %d):' % k,
                  enum._mols_identical(r0.products[k], r1.products[k]))


if __name__ == '__main__':
    main()
