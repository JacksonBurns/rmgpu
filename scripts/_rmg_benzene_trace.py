#!/usr/bin/env python3
"""Get RMG's EXACT benzene+H post-recipe bond orders + aromaticity state (rmg_env)."""
from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.molecule import Molecule


def bonds_of(mol):
    seen = set()
    out = []
    for a in mol.atoms:
        for other, b in a.bonds.items():
            key = (id(a), id(other))
            if key in seen:
                continue
            seen.add(key)
            out.append((a.element.symbol, other.element.symbol, round(b.order, 3), b.is_benzene()))
    return sorted(out)


def main():
    db = KineticsDatabase()
    db.load_families(path=settings['database.directory'] + '/kinetics/families',
                     families=['R_Addition_MultipleBond'])
    fam = db.families['R_Addition_MultipleBond']
    benzene = Molecule().from_smiles('c1ccccc1')
    print('benzene loaded bonds:', bonds_of(benzene))
    h = Molecule().from_smiles('[H]')
    benzene.atoms[0].label = '*1'
    benzene.atoms[1].label = '*2'
    h.atoms[0].label = '*3'
    merged = benzene.merge(h.copy(deep=True))
    print('merged bonds before recipe:', bonds_of(merged))
    fam.forward_recipe.apply_forward(merged)
    print('merged bonds AFTER recipe:', bonds_of(merged))
    print('validAromatic:', merged.props.get('validAromatic'))
    from rmgpy.molecule.kekulize import kekulize
    from rmgpy.exceptions import KekulizationError
    try:
        kekulize(merged)
        print('kekulize OK bonds:', bonds_of(merged))
    except KekulizationError as e:
        print('kekulize KERR; semi bonds:', bonds_of(merged))
    prods = merged.split()
    for i, p in enumerate(prods):
        print('product %d bonds:' % i, bonds_of(p))


if __name__ == '__main__':
    main()
