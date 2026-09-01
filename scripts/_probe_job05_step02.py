#!/usr/bin/env python3
"""Throwaway probe: which (family, reactants) cases work in rmg_env for
job-05/step-02's 30-case reference. Run with rmg_env python."""
import os
import traceback

from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.molecule import Molecule

TEST_FAMS = [
    'R_Recombination', '1,2_shiftC', 'Disproportionation',
    'Singlet_Val6_to_triplet', 'R_Addition_MultipleBond',
    'intra_H_migration', 'Intra_ene_reaction', 'R_Addition_Singlet_Carbene',
]

CASES = [
    # (family, source, [smiles])
    ('H_Abstraction', 'real', ['CC(=O)C', '[H]']),
    ('H_Abstraction', 'real', ['CC', '[OH]']),
    ('H_Abstraction', 'real', ['C', '[CH3]']),
    ('H_Abstraction', 'real', ['C[CH]C', '[H]']),
    ('H_Abstraction', 'real', ['C(=O)[O]', '[H]']),
    ('H_Abstraction', 'real', ['C1=CC=CC=C1', '[H]']),
    ('H_Abstraction', 'real', ['CC(=O)C', 'C[CH]C']),
    ('R_Addition_MultipleBond', 'real', ['C1=CC=CC=C1', '[H]']),
    ('R_Addition_MultipleBond', 'real', ['C=C', '[CH3]']),
    ('R_Addition_MultipleBond', 'real', ['C=C', '[CH2=]']),
    ('intra_H_migration', 'real', ['C[CH]CC']),
    ('intra_H_migration', 'real', ['C[CH]CCC']),
    ('Intra_ene_reaction', 'real', ['C[CH]C=C(C)C']),
    ('R_Addition_RAd', 'real', ['C=C', '[O]O']),
    ('R_Addition_RAd', 'real', ['C=C(C)C', '[OH]']),
    ('R_Recombination', 'test', ['[CH2]', '[CH2]']),
    ('R_Recombination', 'test', ['[OH]', '[OH]']),
    ('R_Recombination', 'test', ['[CH2]', '[OH]']),
    ('R_Addition_Singlet_Carbene', 'real', ['C=C', 'C=O']),
    ('1,2_shiftC', 'test', ['C[CH2]C[CH2]']),
    ('1,2_shiftC', 'test', ['C[CH2][CH2]C(C)(C)C']),
    ('1,2_shiftC', 'test', ['C(C)(C)[CH2]C[CH2]C']),
    ('Singlet_Val6_to_triplet', 'test', ['O=O']),
]


def main():
    real_db = KineticsDatabase()
    real_db.load_families(path=os.path.join(settings['database.directory'],
                                            'kinetics', 'families'),
                          families=[c[0] for c in CASES if c[1] == 'real'])
    test_db = KineticsDatabase()
    test_db.load_families(
        path=os.path.join(settings['test_data.directory'],
                          'testing_database/kinetics/families'),
        families=[c[0] for c in CASES if c[1] == 'test'])

    for fam_label, source, smiles in CASES:
        db = real_db if source == 'real' else test_db
        fam = db.families.get(fam_label)
        if fam is None:
            print('MISSING %-32s %s %s' % (fam_label, source, smiles))
            continue
        try:
            mols = [Molecule().from_smiles(s) for s in smiles]
            rxns = fam.generate_reactions(mols)
            nres = len(mols[0].generate_resonance_structures())
            print('OK      %-32s %s %s  n_rxns=%d res0=%d' %
                  (fam_label, source, ','.join(smiles), len(rxns), nres))
        except Exception as e:
            print('ERROR   %-32s %s %s  %s: %s' %
                  (fam_label, source, ','.join(smiles), type(e).__name__, e))


if __name__ == '__main__':
    main()
