import json
import sys

sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import enumeration as enum

BASE = '/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step02_products_reference.json'


def piece_id_set(m):
    return tuple(sorted(int(a.GetProp('atomid')) for a in m._rdkit.GetAtoms()
                        if a.HasProp('atomid')))


def struct_key(m):
    # label/ID-free structural key
    atoms = []
    for a in m._rdkit.GetAtoms():
        bonds = tuple(sorted(round(b.GetBondTypeAsDouble(), 6)
                             for b in a.GetBonds()))
        atoms.append((a.GetSymbol(), a.GetNumRadicalElectrons(),
                      a.GetFormalCharge(), bonds))
    return tuple(sorted(atoms))


def prod_signature(rxn):
    return tuple(sorted(struct_key(p) for p in rxn.products))


def main():
    data = json.load(open(BASE))
    npass = 0
    nfail = 0
    for case in data['cases']:
        fam = enum.Family.from_reference(case)
        reactants = [Molecule(smiles=s) for s in case['reactant_smiles']]
        fam.matcher = enum.RecordedMatcher(case)
        try:
            rxns = enum.generate_reactions(fam, reactants)
        except Exception as e:
            print('FAIL(exc) %s %s: %s: %s' % (
                case['family'], case['reactant_smiles'],
                type(e).__name__, e))
            nfail += 1
            continue

        got = sorted((prod_signature(r), r.degeneracy) for r in rxns)
        # Compare: number of reactions + per-reaction degeneracy multiset +
        # product structural-signature multiset.
        # For product structural signatures from baseline, parse baseline
        # adjlists into Molecules.
        want_sigs = []
        want_deg = []
        for r in case['reactions']:
            ps = [Molecule.from_adjacency_list(t) for t in r['products']]
            want_sigs.append(tuple(sorted(struct_key(p) for p in ps)))
            want_deg.append(r['degeneracy'])
        got_sig = sorted(prod_signature(r) for r in rxns)
        got_deg = sorted(r.degeneracy for r in rxns)
        ok_sig = got_sig == sorted(want_sigs)
        ok_deg = all(abs(a - b) < 1e-12 for a, b in
                     zip(got_deg, sorted(want_deg)))
        status = 'PASS' if (ok_sig and ok_deg) else 'FAIL'
        if status == 'PASS':
            npass += 1
        else:
            nfail += 1
        print('%s %s %s  (sig=%s deg=%s)' % (
            status, case['family'], case['reactant_smiles'],
            ok_sig, ok_deg))
        if status == 'FAIL':
            print('   got sigs:  %s' % got_sig)
            print('   want sigs: %s' % sorted(want_sigs))
            print('   got deg:   %s' % got_deg)
            print('   want deg:  %s' % sorted(want_deg))
    print('== %d pass, %d fail ==' % (npass, nfail))


if __name__ == '__main__':
    main()
