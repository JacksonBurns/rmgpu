#!/usr/bin/env python3
"""Verify: hook-derived per-application template (via fam.groups.
get_reaction_template on the labeled reactants) == the raw reaction's own
.template. Run in rmg_env. If they match, recording app-level templates in
the capture is sound."""
from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.data.kinetics.common import (
    check_for_same_reactants, ensure_independent_atom_ids, generate_molecule_combos)
from rmgpy.molecule import Molecule


class FakeRxn(object):
    pass


def run(fam, smiles):
    reactants = [Molecule().from_smiles(s) for s in smiles]
    reactants, same = check_for_same_reactants(reactants)
    ensure_independent_atom_ids(reactants, resonance=True)
    app_tmpls = []
    orig = fam._generate_product_structures

    def hook(rs, maps, forward, relabel_atoms=True):
        res = orig(rs, maps, forward, relabel_atoms)
        tmpl = None
        if res is not None:
            try:
                fake = FakeRxn()
                fake.reactants = list(rs)
                tmpl = [e.label for e in fam.groups.get_reaction_template(fake)]
            except Exception as e:
                tmpl = 'ERR:%s:%s' % (type(e).__name__, e)
        app_tmpls.append(tmpl)
        return res

    fam._generate_product_structures = hook
    raw = []
    try:
        for combo in generate_molecule_combos(reactants):
            raw.extend(fam.generate_reactions(list(combo)))
    finally:
        fam._generate_product_structures = orig
    return app_tmpls, raw


def main():
    real = {'intra_H_migration': ['C[CH]CCC'], 'Intra_ene_reaction': ['C[CH]C1=CC=CC=C1']}
    db = KineticsDatabase()
    db.load_families(path=settings['database.directory'] + '/kinetics/families',
                     families=sorted(real))
    for fam_label, smiles in real.items():
        fam = db.families[fam_label]
        app_tmpls, raw = run(fam, smiles)
        print('=' * 60)
        print(fam_label, smiles, 'raw=%d' % len(raw))
        raw_tmpls = [list(r.template) for r in raw]
        ok_app = [t for t in app_tmpls if t is not None]
        print('app tmpl count (ok apps):', len(ok_app), ' raw tmpl count:', len(raw_tmpls))
        match = ok_app == raw_tmpls
        print('app-level templates == raw reaction templates:', match)
        if not match:
            print('  app:', ok_app)
            print('  raw:', raw_tmpls)


if __name__ == '__main__':
    main()
