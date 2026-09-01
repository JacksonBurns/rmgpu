#!/usr/bin/env python3
"""
job-05/step-05 reference capture (job-05 gate ground truth).

Runs RMG-Py's KineticsFamily.generate_reactions (rmg_env) on the full gate
case set (gates/gate05_cases.py) and records ground truth for rmgpu's product
enumeration:

  - family metadata (recipe, reverse recipe, own_reverse, reversible, ...);
  - the reactant SMILES + same_reactants count;
  - the FWD collapsed ground-truth reactions (product adjlists + per-product
    degeneracy + direction + template) - the product SET and degeneracy the
    rmgpu port must reproduce EXACTLY;
  - the REV collapsed ground-truth reactions (RMG's reverse-direction
    generation for non-own-reverse reversible families) - the input to the
    reverse-matching check;
  - calculate_degeneracy on each fwd collapsed reaction (RMG's own method).

The product SMILES are recorded using rmgpy's own canonical SMILES so the
gate can cross-check rmgpu's canonical SMILES.

Non-circular: uses only rmgpy + the RMG databases, never rmgpu code.

Run with: /home/jackson/miniforge3/envs/rmg_env/bin/python
          scripts/record_job05_step05_reference.py
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'gates'))
from gate05_cases import CASES  # noqa: E402

from rmgpy import settings  # noqa: E402
from rmgpy.data.kinetics.database import KineticsDatabase  # noqa: E402
from rmgpy.data.kinetics.common import (  # noqa: E402
    check_for_same_reactants, ensure_independent_atom_ids,
    find_degenerate_reactions, generate_molecule_combos,
)
from rmgpy.molecule import Molecule  # noqa: E402
from rmgpy.species import Species  # noqa: E402


def family_meta(fam):
    fwd = fam.forward_recipe
    rev = fam.reverse_recipe
    eff_fwd = fam.product_num or len(fam.forward_template.products)
    if fam.reverse_template is not None:
        eff_rev = fam.reactant_num or len(fam.reverse_template.products)
    else:
        eff_rev = None
    eff_tr = len(fam.forward_template.reactants)
    if eff_tr == 1:
        item = fam.forward_template.reactants[0].item
        try:
            eff_tr = len(item.split())
        except AttributeError:
            pass
    return {
        'recipe': [list(a) for a in fwd.actions],
        'reverse_recipe': ([list(a) for a in rev.actions]
                           if rev is not None else None),
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
    if isinstance(m, Species):
        return m.molecule[0].to_adjacency_list(remove_h=False)
    return m.to_adjacency_list(remove_h=False)


def smiles_of(m):
    if isinstance(m, Species):
        return m.molecule[0].to_smiles()
    return m.to_smiles()


def rxn_to_dict(rxn):
    return {
        'reactants': [adj_of(r) for r in rxn.reactants],
        'reactant_smiles': [smiles_of(r) for r in rxn.reactants],
        'products': [adj_of(p) for p in rxn.products],
        'product_smiles': [smiles_of(p) for p in rxn.products],
        'degeneracy': rxn.degeneracy,
        'is_forward': bool(rxn.is_forward),
        'reversible': bool(getattr(rxn, 'reversible', False)),
        'template': list(getattr(rxn, 'template', []) or []),
    }


def capture_case(fam, reactant_smiles):
    """Run RMG's generation for one (family, reactants) case, mirroring
    RMG's flow (database.py react_molecules + model loop):
        reactants, same = check_for_same_reactants(reactants)
        ensure_independent_atom_ids(reactants, resonance=True)
        for combo in generate_molecule_combos(reactants):
            fwd_rev = fam.generate_reactions(list(combo))
        fwd = [r for r in fwd_rev if r.is_forward]
        rev = [r for r in fwd_rev if not r.is_forward]
        fwd_c = find_degenerate_reactions(fwd, same, kinetics_family=fam)
        rev_c = find_degenerate_reactions(rev, same, kinetics_family=fam)
    """
    reactants = [Molecule().from_smiles(s) for s in reactant_smiles]
    reactants, same_reactants = check_for_same_reactants(reactants)
    ensure_independent_atom_ids(reactants, resonance=True)

    raw = []
    for combo in generate_molecule_combos(reactants):
        raw.extend(fam.generate_reactions(list(combo)))

    fwd_raw = [r for r in raw if r.is_forward]
    rev_raw = [r for r in raw if not r.is_forward]

    fwd_c = find_degenerate_reactions(fwd_raw, same_reactants=same_reactants,
                                      template=None, kinetics_family=fam,
                                      resonance=True)
    rev_c = find_degenerate_reactions(rev_raw, same_reactants=same_reactants,
                                      template=None, kinetics_family=fam,
                                      resonance=True)

    for r in fwd_c:
        try:
            r._cd = fam.calculate_degeneracy(r)
        except Exception as e:  # noqa: BLE001
            r._cd = None
            r._cd_error = '%s: %s' % (type(e).__name__, e)

    # Reverse-matching ground truth. For an OWN-REVERSE reversible family the
    # reverse reaction (products -> reactants) is the SAME family re-applied to
    # the products: RMG's model loop does exactly this (add_reverse_attribute /
    # calculate_degeneracy on the forward reaction). Re-enumerate each fwd
    # reaction's products (as reactants) and record whether the original
    # reactant set is recovered among the regenerated products. Non-own-reverse
    # families (Disproportionation, R_Recombination, R_Addition_MultipleBond,
    # Singlet_Val6_to_triplet) have a DIFFERENT reverse family
    # (Molecular_Addition / Bond_Dissociation / ...) that is NOT in the
    # 'default' set, so their reverse check is out of scope (recorded None).
    from rmgpy.reaction import same_species_lists
    reverse_checks = []
    for fr in fwd_c:
        if fam.own_reverse and fam.reversible:
            prod_mols = [p.molecule if isinstance(p, Species) else p
                         for p in fr.products]
            try:
                pmols = [p for p in prod_mols]
                pmols, same_p = check_for_same_reactants(pmols)
                ensure_independent_atom_ids(pmols, resonance=True)
                raw_rev = []
                for combo in generate_molecule_combos(pmols):
                    raw_rev.extend(fam.generate_reactions(list(combo)))
                raw_rev = [r for r in raw_rev if r.is_forward]
                reg = find_degenerate_reactions(
                    raw_rev, same_reactants=same_p, template=None,
                    kinetics_family=fam, resonance=True)
                recovers = False
                for rg in reg:
                    try:
                        if same_species_lists(fr.reactants, rg.products,
                                              strict=False):
                            recovers = True
                            break
                    except Exception:  # noqa: BLE001
                        continue
                reverse_checks.append({
                    'reverse_ok': bool(recovers),
                    'reverse_degeneracy': (
                        [r.degeneracy for r in reg if
                         same_species_lists(fr.reactants, r.products,
                                            strict=False)]
                        if recovers else []),
                })
            except Exception as e:  # noqa: BLE001
                reverse_checks.append({
                    'reverse_ok': False,
                    'reverse_degeneracy': [],
                    'error': '%s: %s' % (type(e).__name__, e),
                })
        else:
            reverse_checks.append({'reverse_ok': None})

    return {
        'family': fam.label,
        'reactant_smiles': reactant_smiles,
        'same_reactants': int(same_reactants),
        'family_meta': family_meta(fam),
        'n_fwd_raw': len(fwd_raw),
        'n_rev_raw': len(rev_raw),
        'fwd_reactions': [rxn_to_dict(r) for r in fwd_c],
        'n_fwd_reactions': len(fwd_c),
        'rev_reactions': [rxn_to_dict(r) for r in rev_c],
        'n_rev_reactions': len(rev_c),
        'reverse_checks': reverse_checks,
        'n_fwd_calc': sum(1 for r in fwd_c if getattr(r, '_cd', None) is not None),
        'n_fwd_calc_err': sum(1 for r in fwd_c
                              if getattr(r, '_cd_error', None) is not None),
        'calc_degeneracy': [getattr(r, '_cd', None) for r in fwd_c],
        'calc_degeneracy_error': [getattr(r, '_cd_error', None) for r in fwd_c],
    }


def main():
    real_fams = sorted({c[0] for c in CASES if c[1] == 'default'})
    test_fams = sorted({c[0] for c in CASES if c[1] == 'test'})

    real_db = KineticsDatabase()
    real_db.load_families(
        path=os.path.join(settings['database.directory'], 'kinetics',
                          'families'),
        families=real_fams)
    test_db = KineticsDatabase()
    test_db.load_families(
        path=os.path.join(settings['test_data.directory'],
                          'testing_database/kinetics/families'),
        families=test_fams)

    out = {'cases': []}
    for fam_label, source, smiles in CASES:
        db = real_db if source == 'default' else test_db
        fam = db.families.get(fam_label)
        if fam is None:
            print('MISSING family: %s (%s)' % (fam_label, source))
            continue
        try:
            case = capture_case(fam, smiles)
            case['source'] = source
            out['cases'].append(case)
            print('OK   %-26s %s  fwd=%d rev=%d same=%d' %
                  (fam_label, ','.join(smiles), case['n_fwd_reactions'],
                   case['n_rev_reactions'], case['same_reactants']))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print('ERROR %-26s %s  %s: %s' %
                  (fam_label, ','.join(smiles), type(e).__name__, e))

    out_path = os.path.join(REPO, 'gates', 'baselines', 'job05',
                            'step05_gate_reference.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1, sort_keys=False)
    print('\nwrote', out_path)
    print('cases:', len(out['cases']))
    print('total fwd reactions:', sum(c['n_fwd_reactions']
                                      for c in out['cases']))
    print('total rev reactions:', sum(c['n_rev_reactions']
                                      for c in out['cases']))


if __name__ == '__main__':
    main()
