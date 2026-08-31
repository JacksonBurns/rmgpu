#!/usr/bin/env python3
"""
job-05/step-02 reference capture.

Runs RMG-Py's KineticsFamily.generate_reactions (rmg_env) on a fixed set of
(family, reactants) cases and records ground truth for rmgpu's product
enumeration orchestration:

  - family metadata (recipe, reverse recipe, own_reverse, reversible,
    allow_charged_species, electrons, product_num, reactant_num, reverse_map);
  - the reactant SMILES;
  - "applications": every _generate_product_structures call, captured as the
    LABELED reactant adjacency lists (RMG's own enumeration drives which
    atom-labeling is applied, including resonance isomers and the bimolecular
    A+B / B+A swap branches). This is what rmgpu replays through its own
    apply_recipe - the matcher is a hook (step 03), so step 02 is tested
    against RMG's own labelings, not rmgpu's.
  - "reactions": the COLLAPSED ground truth from find_degenerate_reactions
    (product adjlists + per-product degeneracy + direction + template) - the
    product SET and degeneracy the rmgpu port must reproduce EXACTLY.

Non-circular: uses only rmgpy + the RMG databases, never rmgpu code.

Run with: /home/jackson/miniforge3/envs/rmg_env/bin/python scripts/record_job05_step02_reference.py
"""
import copy
import json
import os

from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.data.kinetics.common import (
    check_for_same_reactants, ensure_independent_atom_ids,
    find_degenerate_reactions, generate_molecule_combos,
)
from rmgpy.molecule import Molecule
from rmgpy.species import Species

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (family, source, [reactant SMILES])
# source: 'real' = RMG-database default families (the ones that drive
# superminimal / c3h4), 'test' = RMG-Py testing_database family.
# Kept to unimolecular / bimolecular gas-phase cases (no surface / termolecular).
CASES = [
    # H abstraction (real family; multiple H sites, resonance isomers)
    ('H_Abstraction', 'real', ['C', '[H]']),
    ('H_Abstraction', 'real', ['CC', '[OH]']),
    ('H_Abstraction', 'real', ['CC(=O)C', '[H]']),
    ('H_Abstraction', 'real', ['C[CH]C', '[H]']),
    ('H_Abstraction', 'real', ['C1=CC=CC=C1', '[H]']),
    ('H_Abstraction', 'real', ['C=C', '[CH3]']),
    ('H_Abstraction', 'real', ['C(C)(C)C', '[CH3]']),
    ('H_Abstraction', 'real', ['CC(=O)[O]', '[H]']),
    ('H_Abstraction', 'real', ['C1CCC1', '[OH]']),
    ('H_Abstraction', 'real', ['C1CC1', '[CH3]']),
    ('H_Abstraction', 'real', ['CCO', '[OH]']),
    ('H_Abstraction', 'real', ['C#C', '[H]']),
    ('H_Abstraction', 'real', ['CCN', '[H]']),
    ('H_Abstraction', 'real', ['CC#C', '[CH3]']),
    # radical-radical recombination (test family; own_reverse=False, reversible)
    ('R_Recombination', 'test', ['[OH]', '[OH]']),
    ('R_Recombination', 'test', ['[CH3]', '[OH]']),
    ('R_Recombination', 'test', ['[CH3]', '[CH3]']),
    # multiple-bond addition (real family; double + triple bonds)
    ('R_Addition_MultipleBond', 'real', ['C=C', '[CH3]']),
    ('R_Addition_MultipleBond', 'real', ['C=C', '[OH]']),
    ('R_Addition_MultipleBond', 'real', ['C1=CC=CC=C1', '[H]']),
    ('R_Addition_MultipleBond', 'real', ['C=O', '[CH3]']),
    ('R_Addition_MultipleBond', 'real', ['C#C', '[H]']),
    ('R_Addition_MultipleBond', 'real', ['C#C', '[CH3]']),
    # intra families (real, self-reverse, label relabeling)
    ('intra_H_migration', 'real', ['C[CH]CCC']),
    ('intra_H_migration', 'real', ['C[CH]C1CCCCC1']),
    ('intra_H_migration', 'real', ['C(C)(C)[CH]C(C)(C)C']),
    ('Intra_ene_reaction', 'real', ['C[CH]C1=CC=CC=C1']),
    # 1,2 shifts (test family; self-reverse)
    ('1,2_shiftC', 'test', ['CC[CH]C']),
    ('1,2_shiftC', 'test', ['CCC[CH]C(C)C']),
    # O2 dissociation (test; irreversible)
    ('Singlet_Val6_to_triplet', 'test', ['O=O']),
]


def family_meta(fam):
    fwd = fam.forward_recipe
    rev = fam.reverse_recipe
    # Effective counts RMG's apply_recipe resolves to
    # (step-01 convention): forward: product_num or len(forward products);
    # reverse: reactant_num or len(reverse_template products)
    # (for self-reverse families the reverse direction is never applied).
    eff_fwd = fam.product_num or len(fam.forward_template.products)
    if fam.reverse_template is not None:
        eff_rev = fam.reactant_num or len(fam.reverse_template.products)
    else:
        eff_rev = None
    # Effective reactant count of the forward template after RMG's
    # single-group split (e.g. R_Recombination's Y_rad group -> 2 reactants).
    eff_tr = len(fam.forward_template.reactants)
    if eff_tr == 1:
        item = fam.forward_template.reactants[0].item
        try:
            eff_tr = len(item.split())
        except AttributeError:
            pass
    return {
        'recipe': [list(a) for a in fwd.actions],
        'reverse_recipe': [list(a) for a in rev.actions] if rev is not None else None,
        'own_reverse': bool(fam.own_reverse),
        'reversible': bool(fam.reversible),
        'allow_charged_species': bool(fam.allow_charged_species),
        'electrons': int(fam.electrons),
        'product_num': fam.product_num,
        'reactant_num': fam.reactant_num,
        'reverse_map': fam.reverse_map,
        'num_template_reactants': len(fam.forward_template.reactants),
        'num_template_reactants_effective': eff_tr,
        'effective_product_num_forward': eff_fwd,
        'effective_product_num_reverse': eff_rev,
    }


def adj_of(m):
    """Adjacency list of a Molecule or Species (no species label prefix)."""
    if isinstance(m, Species):
        # Species.to_adjacency_list prefixes the label; use the bare molecule.
        return m.molecule[0].to_adjacency_list(remove_h=False)
    return m.to_adjacency_list(remove_h=False)


def rxn_to_dict(rxn):
    d = {
        'reactants': [adj_of(r) for r in rxn.reactants],
        'products': [adj_of(p) for p in rxn.products],
        'degeneracy': rxn.degeneracy,
        'is_forward': bool(rxn.is_forward),
        'reversible': bool(getattr(rxn, 'reversible', False)),
        'template': list(getattr(rxn, 'template', []) or []),
    }
    if hasattr(rxn, '_cd'):
        d['calc_degeneracy'] = rxn._cd
        d['calc_degeneracy_error'] = getattr(rxn, '_cd_error', None)
    return d


def capture_case(fam, reactant_smiles):
    """
    Run RMG's reaction generation, mirroring RMG's actual flow
    (rmgpy/data/kinetics/database.py react_molecules + the model loop):

        reactants, same_reactants = check_for_same_reactants(reactants)
        ensure_independent_atom_ids(reactants, resonance=True)   # -> Species
        for combo in generate_molecule_combos(reactants):
            raw += family.generate_reactions(list(combo))
        collapsed = find_degenerate_reactions(raw, same_reactants, ...,
                                              kinetics_family=fam,
                                              resonance=True)

    It captures every labeled reactant application (what rmgpu replays
    through apply_recipe) and the collapsed ground-truth reactions (product
    set + per-product degeneracy).
    """
    reactants = [Molecule().from_smiles(s) for s in reactant_smiles]

    # RMG's actual flow (database.py): same-reactant check + independent
    # atom IDs + resonance expansion BEFORE generate_reactions, then a
    # single find_degenerate_reactions over all combos.
    reactants, same_reactants = check_for_same_reactants(reactants)
    ensure_independent_atom_ids(reactants, resonance=True)

    applications = []
    orig_gen = fam._generate_product_structures

    def wrapper(reactant_structures, maps, forward, relabel_atoms=True):
        # Run the real thing (labels the atoms in place, returns products or None).
        result = orig_gen(reactant_structures, maps, forward, relabel_atoms)
        # After the call the reactant atoms carry this application's labels
        # (or the labels were cleared on a ForbiddenStructureException, in
        # which case `ok` is False and the recorded lists are just the input).
        apps = [r.to_adjacency_list(remove_h=False) for r in reactant_structures]
        # Record RMG's exact per-atom IDs (in the adjacency-list atom order),
        # so the rmgpu replay can reproduce the identical-vs-isomorphic
        # degeneracy distinction (RMG Molecule.is_identical keys on atom.id).
        atom_ids = [[int(atom.id) for atom in r.atoms] for r in reactant_structures]
        applications.append({
            'forward': bool(forward),
            'relabel_atoms': bool(relabel_atoms),
            'labeled_reactants': apps,
            'reactant_atom_ids': atom_ids,
            'ok': result is not None,
        })
        return result

    raw = []
    fam._generate_product_structures = wrapper
    try:
        for combo in generate_molecule_combos(reactants):
            raw.extend(fam.generate_reactions(list(combo)))
    finally:
        fam._generate_product_structures = orig_gen

    # Per-raw-reaction template (RMG sets reaction.template after the
    # products-filter, before degeneracy collapse). Record it in RAW order so
    # the rmgpu replay can re-associate each raw reaction with its template
    # (the template drives the isomorphic-but-different-template duplicate
    # split in find_degenerate_reactions). NOTE: len(raw) can be < len
    # applications, because some applications yield a product identical to the
    # reactant and are dropped by _create_reaction.
    raw_templates = [list(getattr(r, 'template', None) or []) for r in raw]
    assert len(raw_templates) == len(raw), \
        'raw_templates/len mismatch: %d vs %d' % (len(raw_templates), len(raw))

    # Collapsed ground truth (product set + per-product degeneracy).
    collapsed = find_degenerate_reactions(
        raw, same_reactants=same_reactants, template=None,
        kinetics_family=fam, resonance=True)

    # Per-reaction calculate_degeneracy ground truth (RMG's own method,
    # used by the model loop after generation).
    for r in collapsed:
        try:
            r._cd = fam.calculate_degeneracy(r)
        except Exception as e:
            r._cd = None
            r._cd_error = '%s: %s' % (type(e).__name__, e)

    return {
        'family': fam.label,
        'reactant_smiles': reactant_smiles,
        'same_reactants': int(same_reactants),
        'family_meta': family_meta(fam),
        'applications': applications,
        'n_applications': len(applications),
        'n_raw_reactions': len(raw),
        'raw_templates': raw_templates,
        'reactions': [rxn_to_dict(r) for r in collapsed],
        'n_reactions': len(collapsed),
    }


def main():
    real_fams = sorted({c[0] for c in CASES if c[1] == 'real'})
    test_fams = sorted({c[0] for c in CASES if c[1] == 'test'})

    real_db = KineticsDatabase()
    real_db.load_families(
        path=os.path.join(settings['database.directory'], 'kinetics', 'families'),
        families=real_fams)
    test_db = KineticsDatabase()
    test_db.load_families(
        path=os.path.join(settings['test_data.directory'],
                          'testing_database/kinetics/families'),
        families=test_fams)

    out = {'cases': []}
    for fam_label, source, smiles in CASES:
        db = real_db if source == 'real' else test_db
        fam = db.families.get(fam_label)
        if fam is None:
            print('MISSING family: %s (%s)' % (fam_label, source))
            continue
        try:
            case = capture_case(fam, smiles)
            case['source'] = source
            out['cases'].append(case)
            print('OK   %-26s %s  apps=%d raw=%d rxns=%d' %
                  (fam_label, ','.join(smiles), case['n_applications'],
                   case['n_raw_reactions'], case['n_reactions']))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print('ERROR %-26s %s  %s: %s' % (fam_label, ','.join(smiles),
                                               type(e).__name__, e))

    out_path = os.path.join(REPO, 'gates', 'baselines', 'job05',
                            'step02_products_reference.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1, sort_keys=False)
    print('\nwrote', out_path)
    print('cases:', len(out['cases']))
    print('total applications:', sum(c['n_applications'] for c in out['cases']))


if __name__ == '__main__':
    main()
