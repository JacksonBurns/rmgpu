#!/usr/bin/env python3
"""Empirically pin RMG's is_identical(strict=False) semantics on the acetyl
radical resonance pair (same atom-ID set, radical + C=O bond swapped between
the two O). Run in rmg_env."""
from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.data.kinetics.common import (
    check_for_same_reactants, ensure_independent_atom_ids,
    find_degenerate_reactions, generate_molecule_combos)
from rmgpy.molecule import Molecule


def dump(mol, name):
    print('  %s bonds:' % name)
    for b in mol.edges.values() if hasattr(mol, 'edges') else []:
        pass
    for a in mol.atoms:
        nbrs = mol.get_bonds(a)
        print('    atom id=%s %s rad=%d chg=%d  nbrs=%s' % (
            a.id, a.element.symbol, a.radical_electrons, a.charge,
            sorted((o.id, mol.get_bond(a, o).order) for o in nbrs)))


def main():
    db = KineticsDatabase()
    db.load_families(path=settings['database.directory'] + '/kinetics/families',
                     families=['H_Abstraction'])
    fam = db.families['H_Abstraction']
    reactants = [Molecule().from_smiles(s) for s in ['CC(=O)[O]', '[H]']]
    reactants, same = check_for_same_reactants(reactants)
    ensure_independent_atom_ids(reactants, resonance=True)
    raw = []
    for combo in generate_molecule_combos(reactants):
        raw.extend(fam.generate_reactions(list(combo)))
    print('raw=%d' % len(raw))
    a, b = raw[0].products[0], raw[3].products[0]  # the radical product
    dump(a, 'rxn0 prod')
    dump(b, 'rxn3 prod')
    print('rxn0.prod_ids:', tuple(sorted(x.id for x in a.atoms)))
    print('rxn3.prod_ids:', tuple(sorted(x.id for x in b.atoms)))
    print('is_identical strict=False:', a.is_identical(b, strict=False))
    print('is_identical strict=True :', a.is_identical(b, strict=True))
    print('is_isomorphic strict=False:', a.is_isomorphic(b, strict=False))
    print('is_isomorphic strict=True :', a.is_isomorphic(b, strict=True))
    # Now test a pair that differs ONLY in connectivity to see if bonds are checked
    # (build two Molecules with same elements/IDs but different bond between two atoms)
    print('--- connectivity probe ---')
    m1 = Molecule().from_smiles('CC')  # ethane C-C
    m2 = Molecule().from_smiles('CC')
    # give both same ids by default; just confirm equivalent
    print('ethane vs ethane is_identical strict=False (fresh ids):', m1.is_identical(m2, strict=False))


if __name__ == '__main__':
    main()
