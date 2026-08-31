#!/usr/bin/env python3
"""Record RMG-Py's per-application PRODUCTS (with exact atom IDs) for the
job-05/step-02 reference cases, so the rmgpu replay can be diffed
application-by-application. Run with rmg_env python:
/home/jackson/miniforge3/envs/rmg_env/bin/python scripts/capture_step02_app_products.py
Writes gates/baselines/job05/step02_app_products.json (untracked debug aid;
supplements, does not replace, step02_products_reference.json).
"""
import json
import os

from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.data.kinetics.common import (
    check_for_same_reactants, ensure_independent_atom_ids,
    find_degenerate_reactions, generate_molecule_combos,
)
from rmgpy.molecule import Molecule

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(REPO, 'gates', 'baselines', 'job05',
                   'step02_products_reference.json')


def dump_mol(m):
    """Product dump: atoms with exact IDs + bonds (by ID, order)."""
    atoms = []
    for a in m.atoms:
        atoms.append({
            'id': int(a.id),
            'symbol': a.symbol,
            'radical': int(a.radical_electrons),
            'charge': int(a.charge),
        })
    bonds = []
    seen = set()
    for a in m.atoms:
        for other, bond in a.bonds.items():
            i1, i2 = int(a.id), int(other.id)
            key = (min(i1, i2), max(i1, i2))
            if key in seen:
                continue
            seen.add(key)
            order = getattr(bond.order, 'value', bond.order)
            bonds.append((i1, i2, float(order)))
    return {'atoms': atoms, 'bonds': bonds}


def main():
    with open(REF) as f:
        ref = json.load(f)
    fam_labels = sorted({c['family'] for c in ref['cases']})
    real_fams = sorted({l for l, c in zip(fam_labels, ref['cases'])
                        if c.get('source') == 'real'})
    test_fams = sorted({l for l, c in zip(fam_labels, ref['cases'])
                        if c.get('source') == 'test'})
    # map family->source properly
    src = {c['family']: c.get('source') for c in ref['cases']}
    real_fams = sorted({l for l in src if src[l] == 'real'})
    test_fams = sorted({l for l in src if src[l] == 'test'})

    real_db = KineticsDatabase()
    real_db.load_families(
        path=os.path.join(settings['database.directory'], 'kinetics', 'families'),
        families=real_fams)
    test_db = KineticsDatabase()
    test_db.load_families(
        path=os.path.join(settings['test_data.directory'],
                          'testing_database/kinetics/families'),
        families=test_fams)

    out = {}
    for case in ref['cases']:
        fam_label = case['family']
        source = case.get('source', 'real')
        db = real_db if source == 'real' else test_db
        fam = db.families.get(fam_label)
        if fam is None:
            print('MISSING', fam_label)
            continue
        reactants = [Molecule().from_smiles(s) for s in case['reactant_smiles']]
        reactants, same_reactants = check_for_same_reactants(reactants)
        ensure_independent_atom_ids(reactants, resonance=True)

        products_by_app = []
        orig_gen = fam._generate_product_structures

        def wrapper(reactant_structures, maps, forward, relabel_atoms=True):
            result = orig_gen(reactant_structures, maps, forward, relabel_atoms)
            if result is None:
                products_by_app.append(None)
            else:
                products_by_app.append([dump_mol(p) for p in result])
            return result

        fam._generate_product_structures = wrapper
        try:
            raw = []
            for combo in generate_molecule_combos(reactants):
                raw.extend(fam.generate_reactions(list(combo)))
        finally:
            fam._generate_product_structures = orig_gen
        out['%s|%s' % (fam_label, ','.join(case['reactant_smiles']))] = {
            'n_apps': len(products_by_app),
            'products_by_app': products_by_app,
        }
        print('OK %s %s: %d apps, %d non-None' % (
            fam_label, ','.join(case['reactant_smiles']),
            len(products_by_app),
            sum(1 for p in products_by_app if p is not None)))

    out_path = os.path.join(REPO, 'gates', 'baselines', 'job05',
                            'step02_app_products.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
