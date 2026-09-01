import json
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import enumeration as enum

BASE = '/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step02_products_reference.json'


def radical_ids(m):
    return [a.GetProp('atomid') for a in m._rdkit.GetAtoms()
            if a.GetNumRadicalElectrons() > 0 and a.HasProp('atomid')]


def main():
    data = json.load(open(BASE))
    for case in data['cases']:
        if (case['family'], tuple(case['reactant_smiles'])) != \
           ('1,2_shiftC', ('CCC[CH]C(C)C',)):
            continue
        fam = enum.Family.from_reference(case)
        m = enum.RecordedMatcher(case)
        print('per-application products:')
        seen = {}
        for i, structures in enumerate(m.yield_applications()):
            rxn_list = []
            enum._apply_one_application(fam, structures, True, rxn_list)
            if not rxn_list:
                print('  app%d -> no product' % i)
                continue
            p = rxn_list[0].products[0]
            ids = sorted(int(a.GetProp('atomid'))
                         for a in p._rdkit.GetAtoms() if a.HasProp('atomid'))
            rad = radical_ids(p)
            key = (Chem.MolToSmiles(Chem.Mol(p._rdkit)), tuple(ids), tuple(sorted(rad)))
            seen.setdefault(key, []).append(i)
            print('  app%d ids=%s rad=%s' % (i, ids, rad))
        print('distinct (smiles,idset,rad) groups:', len(seen))
        for k, apps in seen.items():
            print('   n=%d apps=%s smiles=%s' % (len(apps), apps, k[0][:40]))
        # now check my identical pairs
        print('my identical pairs among products:')
        prods = []
        for structures in m.yield_applications():
            rxn_list = []
            enum._apply_one_application(fam, structures, True, rxn_list)
            prods.append(rxn_list[0].products[0] if rxn_list else None)
        valid = [(i, p) for i, p in enumerate(prods) if p is not None]
        for ia, pa in valid:
            for ib, pb in valid:
                if ia >= ib:
                    continue
                same = enum._mols_identical(pa, pb)
                if same:
                    print('  identical: app%d ~ app%d' % (ia, ib))


if __name__ == '__main__':
    main()
