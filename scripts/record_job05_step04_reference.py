#!/usr/bin/env python3
"""
job-05/step-04 reference capture (RMG-Py, rmg_env).

Ground truth for the step-04 family loader + KineticsFamilies facade:

  1. For EVERY family in the 'default' recommended set (RMG-database/input/
     kinetics/families/recommended.py): the loaded KineticsFamily's
     enumeration-relevant fields - template reactant/product labels, group
     tree (entries + parent/children + top, in RMG's load order), recipe /
     reverse recipe, own_reverse / reversible / allow_charged_species /
     electrons / reactant_num / product_num / auto_generated / reverse_map,
     the EFFECTIVE reactant count after RMG's single-group split
     (item.split()), and the effective forward product count - recorded the
     way RMG's own load + family_meta semantics resolve them (non-circular:
     the rmgpu loader is checked against this, not the other way round).

  2. match_reaction ground truth: 20 (family, reaction) pairs drawn from the
     step-03 verdict set (the reactions RMG generated from the 7 mechanism
     families) plus hand-built negatives. For each pair, RMG's own
     verdict - run EVERY default family's
     get_labeled_reactants_and_products + get_reaction_template_labels
     (the forward template-match + recipe + product check + tree descent) -
     recording (matched, template_labels, error). The facade's
     match_reaction(reaction) is checked against these: it must attribute
     the reaction to the family(ies) RMG says match it, with the labels RMG
     says they are.

The reaction structures come from the step-02 reference (recorded RMG
products), so both sides compare identical molecules.

Run: /home/jackson/miniforge3/envs/rmg_env/bin/python scripts/record_job05_step04_reference.py
"""
import json
import os

from rmgpy import settings
from rmgpy.data.base import LogicNode
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.molecule import Molecule
from rmgpy.reaction import Reaction

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP02_REF = os.path.join(REPO, 'gates', 'baselines', 'job05',
                          'step02_products_reference.json')
FAMILIES_PATH = settings['database.directory'] + '/kinetics/families'


def _default_set():
    """The 'default' recommended set, read from RMG-database's
    families/recommended.py (exec'd - same source of truth RMG-Py uses)."""
    import importlib
    loader = importlib.machinery.SourceFileLoader('rec',
        os.path.join(FAMILIES_PATH, 'recommended.py'))
    spec = importlib.util.spec_from_file_location('rec',
        os.path.join(FAMILIES_PATH, 'recommended.py'), loader=loader)
    rec = importlib.util.module_from_spec(spec)
    loader.exec_module(rec)
    return sorted(rec.__dict__.get('default', set()))


def _family_dump(fam):
    """One family's load state, in RMG's own representation."""
    fwd = fam.forward_template
    # effective reactant count after the single-group split (RMG
    # family_meta convention: item.split() when one template reactant)
    eff_tr = len(fwd.reactants)
    if eff_tr == 1:
        item = fwd.reactants[0].item
        try:
            eff_tr = len(item.split())
        except AttributeError:
            pass
    entries = []
    for label, entry in fam.groups.entries.items():
        node = {
            'label': label,
            'parent': entry.parent.label if entry.parent is not None else None,
            'children': [c.label for c in entry.children],
            'is_logic': isinstance(entry.item, LogicNode),
        }
        if isinstance(entry.item, LogicNode):
            node['symbol'] = entry.item.symbol
            node['invert'] = entry.item.invert
            node['components'] = list(entry.item.components)
        else:
            node['adj'] = entry.item.to_adjacency_list()
        entries.append(node)
    return {
        'name': fam.label,
        'template_reactants': [e.label for e in fwd.reactants],
        'template_products': [e.label for e in fwd.products],
        'num_template_reactants': len(fwd.reactants),
        'num_template_reactants_effective': eff_tr,
        'effective_product_num_forward':
            fam.product_num or len(fwd.products),
        'recipe': [list(a) for a in fam.forward_recipe.actions],
        'reverse_recipe': ([list(a) for a in fam.reverse_recipe.actions]
                           if fam.reverse_recipe is not None else None),
        'own_reverse': bool(fam.own_reverse),
        'reversible': bool(fam.reversible),
        'allow_charged_species': bool(fam.allow_charged_species),
        'electrons': int(fam.electrons),
        'reactant_num': fam.reactant_num,
        'product_num': fam.product_num,
        'auto_generated': bool(fam.auto_generated),
        'reverse_map': fam.reverse_map,
        'num_groups': len(fam.groups.entries),
        'num_groups_file': _count_file_entries(
            os.path.join(FAMILIES_PATH, fam.label, 'groups.py')),
        'top': [e.label for e in fam.groups.top],
        'entries': entries,
    }


def _count_file_entries(path):
    """Number of entry(...) calls in a groups.py (the file-level entry
    count - what the controlled parse sees; RMG's loaded entry dict can
    be larger, it gains generated product-template entries)."""
    import re as _re
    return len(_re.findall(r'^entry\(', open(path).read(), _re.M))


def _mols(adjlists):
    return [Molecule().from_adjacency_list(a) for a in adjlists]


def _verdict(fam, reactants, products):
    """RMG's own forward verdict for (family, reaction): matched +
    most-specific template labels (or the exception text)."""
    try:
        labeled = fam.get_labeled_reactants_and_products(reactants, products)
    except Exception as e:
        return {'matched': False, 'template': None,
                'error': '%s: %s' % (type(e).__name__, e)}
    if labeled[0] is None:
        return {'matched': False, 'template': None, 'error': None}
    try:
        rxn = Reaction(reactants=labeled[0],
                       products=labeled[1] if labeled[1] else products)
        labels = fam.get_reaction_template_labels(rxn)
        return {'matched': True, 'template': list(labels), 'error': None}
    except Exception as e:
        return {'matched': True, 'template': None,
                'error': '%s: %s' % (type(e).__name__, e)}


def main():
    default = _default_set()
    print('default set: %d families' % len(default))
    db = KineticsDatabase()
    db.load_families(path=FAMILIES_PATH, families=default)
    missing = [f for f in default if f not in db.families]
    assert not missing, 'families not loaded: %s' % missing

    fams = {}
    for name in default:
        fams[name] = _family_dump(db.families[name])

    # ---- match_reaction cases (the 20 the facade must reproduce) ----
    ref = json.load(open(STEP02_REF))
    # (case_index, reaction_index) selections: coverage across all 7
    # mechanism families, including the R_Recombination single-group split
    # and the own-reverse H_Abstraction relabeling.
    SELECT = [
        (0, 0),    # H_Abstraction CH4 + H
        (1, 0),    # H_Abstraction CH3CH2 + OH
        (2, 0),    # H_Abstraction acetone + H
        (4, 0),    # H_Abstraction benzene + H (aromatic)
        (6, 1),    # H_Abstraction neopentyl (degeneracy 9)
        (8, 0),    # H_Abstraction cyclopentane + OH
        (11, 0),   # H_Abstraction acetylene + H
        (14, 0),   # R_Recombination OH + OH (single-group split)
        (15, 0),   # R_Recombination CH3 + OH
        (17, 0),   # R_Addition_MultipleBond ethylene + CH3
        (19, 0),   # R_Addition_MultipleBond benzene + H (aromatic)
        (20, 0),   # R_Addition_MultipleBond formaldehyde + CH3
        (21, 0),   # R_Addition_MultipleBond acetylene + H
        (23, 0),   # intra_H_migration 2-methylbutyl
        (24, 1),   # intra_H_migration cyclohexylmethyl
        (25, 1),   # intra_H_migration 2,2,4,4-tetramethylpentyl
        (26, 0),   # Intra_ene_reaction 3-methylallyl benzene
        (27, 0),   # 1,2_shiftC 2-pentyl
        (28, 0),   # 1,2_shiftC 2,3-dimethylbutyl
        (29, 0),   # Singlet_Val6_to_triplet O2
    ]
    reactions = []
    for ci, ri in SELECT:
        case = ref['cases'][ci]
        rxn = case['reactions'][ri]
        reactions.append({
            'ci': ci, 'ri': ri, 'own': case['family'],
            'reactants': rxn['reactants'],
            'products': rxn['products'],
        })
    # Build the reference: for each of the 20 reactions, the verdict of
    # EVERY default family (the facade's match_reaction must return the
    # matching family + labels).
    verdicts = {}
    for rec in reactions:
        key = '%s|%s|%s' % (rec['own'], rec['ci'], rec['ri'])
        reactants = _mols(rec['reactants'])
        products = _mols(rec['products'])
        row = {}
        for name in default:
            row[name] = _verdict(db.families[name], reactants, products)
        verdicts[key] = row

    out = {
        'default': default,
        'families': {n: fams[n] for n in default},
        'match_cases': [
            {'key': '%s|%s|%s' % (r['own'], r['ci'], r['ri']),
             'own': r['own'], 'ci': r['ci'], 'ri': r['ri'],
             'reactant_smiles': None}
            for r in reactions
        ],
        'reactions': reactions,
        'verdicts': verdicts,
    }
    path = os.path.join(REPO, 'gates', 'baselines', 'job05',
                        'step04_families_reference.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', path)
    print('families recorded:', len(fams))
    total_groups = sum(f['num_groups'] for f in fams.values())
    print('total group entries (incl. generated product entries):',
          total_groups)
    # per-case summary
    for rec in reactions:
        key = '%s|%s|%s' % (rec['own'], rec['ci'], rec['ri'])
        matchers = [n for n in default
                    if verdicts[key][n]['matched']]
        print('  %-55s matchers=%s' % (key, matchers))


if __name__ == '__main__':
    main()
