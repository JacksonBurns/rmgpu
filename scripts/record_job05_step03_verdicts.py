#!/usr/bin/env python3
"""
job-05/step-03 verdict capture (RMG-Py, rmg_env).

Ground truth for the rmgpu `match(family, reaction)` entry point, driven off
the step-02 product-enumeration set (gates/baselines/job05/
step02_products_reference.json) - the reactions RMG actually generated.

For each (family, reaction) pair - family in the 7 mechanism families,
reaction in the step-02 generated reactions - record whether RMG's
get_labeled_reactants_and_products(reactants, products) (the FORWARD
template-match + recipe-apply + product check) SUCCEEDS. This is the
faithful "does family F match reaction R" verdict:
  - R's own (generating) family: SUCCEEDS (round-trip), and the most-specific
    template labels (get_reaction_template_labels) are recorded - these are
    what rmgpu match(family, reaction) must return.
  - a WRONG family: usually FAILS (returns None) -> the test's negative cases
    are ground-truthed here, not assumed.

The reactants/products are the step-02 recorded base (unlabeled) molecules.

Run: /home/jackson/miniforge3/envs/rmg_env/bin/python scripts/record_job05_step03_verdicts.py
"""
import json
import os

from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.molecule import Molecule
from rmgpy.reaction import Reaction

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP02_REF = os.path.join(REPO, 'gates', 'baselines', 'job05',
                          'step02_products_reference.json')

FAMILIES = {
    'real': ['H_Abstraction', 'R_Addition_MultipleBond',
             'intra_H_migration', 'Intra_ene_reaction'],
    'test': ['R_Recombination', '1,2_shiftC', 'Singlet_Val6_to_triplet'],
}


def _load_families():
    objs = {}
    real_db = KineticsDatabase()
    real_db.load_families(
        path=settings['database.directory'] + '/kinetics/families',
        families=FAMILIES['real'])
    test_db = KineticsDatabase()
    test_db.load_families(
        path=settings['test_data.directory'] + '/testing_database/kinetics/families',
        families=FAMILIES['test'])
    for db in (real_db, test_db):
        for f in db.families:
            objs[f] = db.families[f]
    return objs


def _mols(adjlists):
    return [Molecule().from_adjacency_list(a) for a in adjlists]


def _verdict_and_labels(fam, rxn):
    """Run RMG's forward template-match + recipe + product check for rxn in
    fam. Return (matched: bool, template_labels: list|None, error: str|None)."""
    reactants = _mols(rxn['reactants'])
    products = _mols(rxn['products'])
    try:
        labeled = fam.get_labeled_reactants_and_products(reactants, products)
    except Exception as e:
        return False, None, '%s: %s' % (type(e).__name__, e)
    if labeled[0] is None:
        return False, None, None
    try:
        rxn2 = Reaction(reactants=labeled[0],
                        products=labeled[1] if labeled[1] else products)
        labels = fam.get_reaction_template_labels(rxn2)
        return True, labels, None
    except Exception as e:
        return True, None, '%s: %s' % (type(e).__name__, e)


def main():
    fams = _load_families()
    ref = json.load(open(STEP02_REF))
    family_list = sorted(fams)

    rxn_list = []   # flat list of (case_index, reaction_index, rxn_dict)
    for ci, case in enumerate(ref['cases']):
        for ri, rxn in enumerate(case['reactions']):
            rxn_list.append((ci, ri, case, rxn))

    out = {
        'families': family_list,
        'reactions': [],
        'verdicts': {},      # key: 'family|ci|ri' -> {matched, template, error}
    }
    n_match = 0
    n_neg = 0
    for ci, ri, case, rxn in rxn_list:
        own = case['family']
        out['reactions'].append({
            'ci': ci, 'ri': ri, 'family': own,
            'reactants': case['reactant_smiles'],
            'template': rxn['template'],
        })
        for fl in family_list:
            matched, labels, err = _verdict_and_labels(fams[fl], rxn)
            out['verdicts']['%s|%d|%d' % (fl, ci, ri)] = {
                'matched': matched, 'template': labels, 'error': err,
            }
            if fl == own and not matched:
                print('WARN: own family %s did NOT match its reaction %s'
                      % (fl, case['reactant_smiles']))
        # tally
        for fl in family_list:
            v = out['verdicts']['%s|%d|%d' % (fl, ci, ri)]
            if v['matched']:
                n_match += 1
            else:
                n_neg += 1

    path = os.path.join(REPO, 'gates', 'baselines', 'job05',
                        'step03_match_verdicts.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', path)
    print('families:', family_list)
    print('reactions:', len(rxn_list))
    print('total verdicts:', len(out['verdicts']),
          ' matched:', n_match, ' not-matched:', n_neg)
    # per-reaction match pattern (which families match each reaction)
    for rec in out['reactions']:
        ci, ri = rec['ci'], rec['ri']
        matchers = [fl for fl in family_list
                    if out['verdicts']['%s|%d|%d' % (fl, ci, ri)]['matched']]
        ownok = rec['family'] in matchers
        print('  ci=%d ri=%d own=%-26s matches=%s%s' % (
            ci, ri, rec['family'], matchers,
            '' if ownok else '  *** OWN FAMILY MISSING ***'))


if __name__ == '__main__':
    main()
