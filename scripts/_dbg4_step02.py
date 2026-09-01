#!/usr/bin/env python3
"""Debug probe: for failing cases, dump each raw rmgpu reaction's product
atom-ID sets + structure, and pairwise identical/isomorphic relations, to see
why rmgpu overcounts vs RMG's recorded degeneracy."""
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
    try:
        return Chem.MolToSmiles(m._rdkit, canonical=True)
    except Exception as e:
        return '<canonical-fail: %s>' % type(e).__name__


def main():
    targets = sys.argv[1:]
    with open(BASE) as f:
        data = json.load(f)
    for case in data['cases']:
        key = case['family'] + '|' + ','.join(case['reactant_smiles'])
        if targets and key not in targets:
            continue
        print('=' * 70)
        print(key, ' same_reactants=%s' % case['same_reactants'])
        for s in case['reactant_smiles']:
            m = Molecule(smiles=s)
            from rmgpu.molecule.resonance import generate_resonance_structures
            try:
                forms = generate_resonance_structures(m)
                print('resonance %s -> %d forms' % (s, len(forms)))
            except Exception as e:
                print('resonance %s -> EXC %s: %s' % (s, type(e).__name__, e))
        fam = enum.Family.from_reference(case)
        match = enum.RecordedMatcher(case)
        rxn_list = []
        for i, structures in enumerate(match.yield_applications()):
            enum._apply_one_application(fam, structures, True, rxn_list)
        print('raw reactions: %d (recorded %d)' % (len(rxn_list), case['n_raw_reactions']))
        for j, r in enumerate(rxn_list):
            print('  rxn %d: %s ids=%s' % (j, [canon(x) for x in r.products],
                                            [enum.piece_atom_ids(x) for x in r.products]))
        print('pairwise (i<j): isomorph/identical')
        for i in range(len(rxn_list)):
            for j in range(i + 1, len(rxn_list)):
                ri, rj = rxn_list[i], rxn_list[j]
                iso = enum.same_species_lists(ri.products, rj.products, strict=False)
                idn = enum.identical_species_lists(ri.products, rj.products)
                if iso or idn:
                    print('  (%d,%d): iso=%s iden=%s' % (i, j, iso, idn))
        print('recorded collapsed:')
        for r in case['reactions']:
            prods = [Molecule.from_adjacency_list(t) for t in r['products']]
            print('  deg=%s %s ids=%s' % (r['degeneracy'],
                                          [canon(p) for p in prods],
                                          [enum.piece_atom_ids(p) for p in prods]))


if __name__ == '__main__':
    main()
