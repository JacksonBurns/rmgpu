#!/usr/bin/env python3
"""Verify rmgpu step-02 product enumeration vs the recorded RMG-Py reference.
Product signatures use canonical SMILES (isomorphism-correct, aromatic-aware).
Run: /home/jackson/miniforge3/envs/rmgpu/bin/python scripts/_verify_step02.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import enumeration as enum

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'gates', 'baselines', 'job05', 'step02_products_reference.json')


def canon(m):
    return Chem.MolToSmiles(m._rdkit, canonical=True)


def prod_sig(rxn):
    """Order-independent product signature via canonical SMILES."""
    return tuple(sorted(canon(p) for p in rxn.products))


def main():
    with open(BASE) as f:
        data = json.load(f)
    npass = nfail = 0
    detail = []
    for case in data['cases']:
        fam = enum.Family.from_reference(case)
        reactants = [Molecule(smiles=s) for s in case['reactant_smiles']]
        fam.matcher = enum.RecordedMatcher(case)
        try:
            rxns = enum.generate_reactions(fam, reactants)
        except Exception as e:
            print('FAIL(exc) %s %s: %s: %s' % (
                case['family'], case['reactant_smiles'], type(e).__name__, e))
            nfail += 1
            detail.append((case, None, 'EXC %s: %s' % (type(e).__name__, e)))
            continue

        got = sorted((prod_sig(r), float(r.degeneracy)) for r in rxns)
        want = sorted((tuple(sorted(canon(Molecule.from_adjacency_list(t))
                                  for t in r['products'])),
                       float(r['degeneracy']))
                      for r in case['reactions'])
        if got == want:
            npass += 1
            print('PASS %-24s %-24s rxns=%d' % (
                case['family'], ','.join(case['reactant_smiles']), len(rxns)))
        else:
            nfail += 1
            print('FAIL %-24s %-24s' % (
                case['family'], ','.join(case['reactant_smiles'])))
            print('   got : %s' % got)
            print('   want: %s' % want)
            detail.append((case, rxns, None))
    print('\n== %d pass, %d fail (of %d) ==' % (npass, nfail, npass + nfail))
    return 0 if nfail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
