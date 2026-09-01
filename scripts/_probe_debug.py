import json
import sys

sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import enumeration as enum

BASE = '/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step02_products_reference.json'


def canonical(m):
    from rdkit import Chem
    try:
        return Chem.MolToSmiles(Chem.Mol(m._rdkit))
    except Exception:
        return '<unsanitized>'


def main():
    data = json.load(open(BASE))
    targets = {
        ('H_Abstraction', ('C1=CC=CC=C1', '[H]')),
        ('H_Abstraction', ('CC(=O)[O]', '[H]')),
        ('intra_H_migration', ('C[CH]CCC',)),
        ('intra_H_migration', ('C[CH]C1CCCCC1',)),
        ('Intra_ene_reaction', ('C[CH]C1=CC=CC=C1',)),
        ('1,2_shiftC', ('CCC[CH]C(C)C',)),
    }
    for case in data['cases']:
        key = (case['family'], tuple(case['reactant_smiles']))
        if key not in targets:
            continue
        print('====', key)
        fam = enum.Family.from_reference(case)
        reactants = [Molecule(smiles=s) for s in case['reactant_smiles']]
        fam.matcher = enum.RecordedMatcher(case)

        # raw
        rxn_list = []
        for structures in fam.matcher.yield_applications():
            if len(structures) != fam.reactant_num_effective:
                continue
            enum._apply_one_application(fam, structures, True, rxn_list)
        print('  raw apps ok:', len(rxn_list),
              ' (recorded n_applications:', case['n_applications'], ')')
        same = enum.check_for_same_reactants(enum.expand_resonance(reactants))
        # groups
        groups = []
        for rxn0 in rxn_list:
            placed = False
            for sub in groups:
                iso = enum.same_species_lists(rxn0.products, sub[0].products,
                                              strict=False)
                if not iso:
                    continue
                ident = enum.identical_species_lists(rxn0.products,
                                                     sub[0].products)
                if ident:
                    placed = True
                    break
                sub.append(rxn0)
                placed = True
                break
            if not placed:
                groups.append([rxn0])
        print('  groups:', len(groups))
        for g in groups:
            p = g[0].products[0]
            print('    group len', len(g),
                  'canonical:', canonical(p))
        collapsed = enum.find_degenerate_reactions(
            rxn_list, same_reactants=same, template=None,
            family=fam, resonance=True)
        print('  collapsed:', [(canonical(r.products[0])
                                if len(r.products) == 1 else
                                tuple(sorted(canonical(p) for p in r.products)),
                                r.degeneracy) for r in collapsed])
        print('  expected :', [(
            [Molecule.from_adjacency_list(t).to_smiles()
             for t in r['products']] if False else
            tuple(sorted(canonical(Molecule.from_adjacency_list(t))
                         for t in r['products'])),
            r['degeneracy']) for r in case['reactions']])


if __name__ == '__main__':
    main()
