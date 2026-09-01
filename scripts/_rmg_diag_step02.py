#!/usr/bin/env python3
"""RMG-Py ground-truth diagnostic (runs in rmg_env).
For named (family|reactants) cases, run RMG's generate_reactions, then for
each pair of RAW reactions dump: product atom-ID sets, isomorphic (strict=False,
check_template_rxn_products), identical (check_identical=True, strict=False,
check_template_rxn_products), and templates. Then run find_degenerate_reactions
and print the collapsed degeneracies. This is the ground truth rmgpu must match."""
import sys
from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.data.kinetics.common import (
    check_for_same_reactants, ensure_independent_atom_ids,
    find_degenerate_reactions, generate_molecule_combos)
from rmgpy.molecule import Molecule

CASES = {
 'H_Abstraction|CC(=O)[O],[H]': ('H_Abstraction', 'real', ['CC(=O)[O]', '[H]']),
 'intra_H_migration|C[CH]CCC': ('intra_H_migration', 'real', ['C[CH]CCC']),
 'Intra_ene_reaction|C[CH]C1=CC=CC=C1': ('Intra_ene_reaction', 'real', ['C[CH]C1=CC=CC=C1']),
 '1,2_shiftC|CCC[CH]C(C)C': ('1,2_shiftC', 'test', ['CCC[CH]C(C)C']),
}


def ids_of(mol):
    m = mol
    # Products may be Species (after ensure_species) or Molecule.
    if hasattr(m, 'molecule'):
        m = m.molecule[0]
    return tuple(sorted(int(a.id) for a in m.atoms))


def prod_ids(rxn):
    return tuple(ids_of(p) for p in rxn.products)


def main():
    want = sys.argv[1:]
    real_fams, test_fams = set(), set()
    for k in (want or list(CASES)):
        fam, src, _ = CASES[k]
        (real_fams if src == 'real' else test_fams).add(fam)
    dbs = {}
    if real_fams:
        db = KineticsDatabase()
        db.load_families(path=settings['database.directory'] + '/kinetics/families',
                         families=sorted(real_fams))
        dbs['real'] = db
    if test_fams:
        db = KineticsDatabase()
        db.load_families(path=settings['test_data.directory'] +
                         '/testing_database/kinetics/families', families=sorted(test_fams))
        dbs['test'] = db

    for k in (want or list(CASES)):
        fam_label, src, smiles = CASES[k]
        fam = dbs[src].families[fam_label]
        reactants = [Molecule().from_smiles(s) for s in smiles]
        reactants, same_reactants = check_for_same_reactants(reactants)
        ensure_independent_atom_ids(reactants, resonance=True)
        raw = []
        for combo in generate_molecule_combos(reactants):
            raw.extend(fam.generate_reactions(list(combo)))
        print('=' * 72)
        print(k, ' raw=%d same_reactants=%s' % (len(raw), same_reactants))
        for i, r in enumerate(raw):
            print('  rxn %d: template=%s prod_ids=%s' % (i, r.template, prod_ids(r)))
        # pairwise
        print('  pairwise (i<j) iso/identical (check_template_rxn_products, strict=False):')
        for i in range(len(raw)):
            for j in range(i + 1, len(raw)):
                a, b = raw[i], raw[j]
                iso = a.is_isomorphic(b, check_identical=False, strict=False,
                                      check_template_rxn_products=True)
                if not iso:
                    continue
                ident = a.is_isomorphic(b, check_identical=True, strict=False,
                                        check_template_rxn_products=True)
                print('    (%d,%d) ident=%s tmpl_eq=%s' % (
                    i, j, ident, frozenset(a.template) == frozenset(b.template)))
        collapsed = find_degenerate_reactions(raw, same_reactants=same_reactants,
                                              template=None, kinetics_family=fam,
                                              resonance=True)
        print('  collapsed: %s' % [(prod_ids(r), r.degeneracy, r.template, r.duplicate)
                                    for r in collapsed])


if __name__ == '__main__':
    main()
