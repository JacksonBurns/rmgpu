#!/usr/bin/env python3
"""Definitive RMG ground-truth diagnostic (rmg_env). For one case, dump each
raw reaction's product (canonical SMILES + per-atom element/radical/charge +
bond orders + IDs) and the full pairwise is_identical / is_isomorphic(strict=False)
matrix as RMG sees it, plus the collapsed result. Removes all ambiguity about
how RMG groups them."""
import sys
from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.data.kinetics.common import (
    check_for_same_reactants, ensure_independent_atom_ids,
    find_degenerate_reactions, generate_molecule_combos)
from rmgpy.molecule import Molecule

CASES = {
 'intra_H_migration|C[CH]CCC': ('intra_H_migration', 'real', ['C[CH]CCC']),
 '1,2_shiftC|CCC[CH]C(C)C': ('1,2_shiftC', 'test', ['CCC[CH]C(C)C']),
 'H_Abstraction|CC(=O)[O],[H]': ('H_Abstraction', 'real', ['CC(=O)[O]', '[H]']),
}


def prod_of(rxn):
    # product[0]; for bimolecular there can be 2 - return tuple of molecules
    prods = rxn.products
    return prods[0] if isinstance(prods[0], Molecule) else prods[0].molecule[0]


def main():
    real_fams, test_fams = set(), set()
    for k in CASES:
        f, s, _ = CASES[k]
        (real_fams if s == 'real' else test_fams).add(f)
    dbs = {}
    if real_fams:
        db = KineticsDatabase(); db.load_families(path=settings['database.directory'] + '/kinetics/families', families=sorted(real_fams)); dbs['real'] = db
    if test_fams:
        db = KineticsDatabase(); db.load_families(path=settings['test_data.directory'] + '/testing_database/kinetics/families', families=sorted(test_fams)); dbs['test'] = db

    for k in CASES:
        fam_label, src, smiles = CASES[k]
        fam = dbs[src].families[fam_label]
        reactants = [Molecule().from_smiles(s) for s in smiles]
        reactants, same = check_for_same_reactants(reactants)
        ensure_independent_atom_ids(reactants, resonance=True)
        raw = []
        for combo in generate_molecule_combos(reactants):
            raw.extend(fam.generate_reactions(list(combo)))
        print('=' * 78)
        print(k, ' raw=%d same_reactants=%s' % (len(raw), same))
        # per-raw product summary
        for i, r in enumerate(raw):
            p = r.products[0]
            m = p.molecule[0] if hasattr(p, 'molecule') else p
            atoms = ' '.join('%d:%s/r%d/c%d' % (a.id, a.element.symbol, a.radical_electrons, a.charge) for a in m.atoms)
            print('  rxn%d tmpl=%s' % (i, r.template))
            print('        atoms: %s' % atoms)
        print('  pairwise is_identical(strict=False) / is_isomorphic(strict=False):')
        n = len(raw)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = raw[i].products[0], raw[j].products[0]
                ia = a.molecule[0] if hasattr(a, 'molecule') else a
                ib = b.molecule[0] if hasattr(b, 'molecule') else b
                iso = ia.is_isomorphic(ib, strict=False)
                ident = ia.is_identical(ib, strict=False)
                print('    (%d,%d) iso=%s ident=%s' % (i, j, iso, ident))
        collapsed = find_degenerate_reactions(raw, same_reactants=same, template=None, kinetics_family=fam, resonance=True)
        print('  collapsed:')
        for r in collapsed:
            m = r.products[0]
            mm = m.molecule[0] if hasattr(m, 'molecule') else m
            print('    deg=%.1f tmpl=%s n_atoms=%d' % (r.degeneracy, r.template, len(mm.atoms)))


if __name__ == '__main__':
    main()
