"""Job-05 gate: product-enumeration parity vs RMG-Py (job-05/step-05).

Case set: gates/gate05_cases.py - every family in the 'default' set that
appears in the c3h4/superminimal mechanisms (species CH2, C2H2, H2, O2, N2)
plus the RMG-Py test fixtures recorded in step 02/03 (32 (family, reactants)
cases, unimolecular + bimolecular gas-phase only).

Checks (the job-05 gate per prompts/steps/job-05-step-05-gate.md):
  1. Product enumeration parity: for each case, rmgpu's
     `generate_reactions` (the fresh path - step-03 group matcher, real
     recipe application, real degeneracy collapse) vs RMG-Py's recorded
     family generation (gates/baselines/job05/step05_gate_reference.json,
     captured by scripts/record_job05_step05_reference.py, which runs
     rmgpy's own KineticsFamily.generate_reactions). The SETS of products
     (canonical SMILES, explicit-H kekulized) must be equal and the
     degeneracy per product must match EXACTLY.
  2. Reverse: for each fwd reaction of an own-reverse reversible family,
     re-enumerate its products through the same family and check the
     original reactant set is recovered among the regenerated products.
     (Non-own-reverse families - Disproportionation, R_Recombination,
     R_Addition_MultipleBond - have their reverse in a DIFFERENT family
     not in the 'default' set; Singlet_Val6_to_triplet is irreversible.
     Their reverse check is out of scope for this gate, recorded None.)
  3. Timing: product enumeration per case recorded; the MAXIMUM must stay
     under 5 s (the case set's largest reactants are 25 atoms explicit,
     well beyond the brief's 10-atom requirement, so the floor is
     stronger than required).

Also records the blocked-families list (the 'default' set minus what the
loader could express) for job 06.

Exit code 0 = GREEN (product sets + degeneracies exact on every case,
reverse recovery holds where in scope, timing under floor), 1 = RED with
the mismatches listed. Writes reports/gate_05_results.json.
"""
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'gates'))

from rmgpu.molecule.molecule import Molecule  # noqa: E402
from rmgpu.core.family import KineticsFamilies, Family  # noqa: E402
from rmgpu.core.template import TemplateMatcher  # noqa: E402
from rmgpu.core import enumeration as enum  # noqa: E402
from gate05_cases import CASES, TIMING_SECONDS  # noqa: E402

REFERENCE = os.path.join(REPO, 'gates', 'baselines', 'job05',
                         'step05_gate_reference.json')
RESULTS = os.path.join(REPO, 'reports', 'gate_05_results.json')

# Families that exist only in RMG-Py's testing_database (not in rmgdb);
# loaded directly from the test data dir (no rmgdb cross-check).
TEST_FAM_DIR = os.path.join(
    '/home/jackson/rmgpu/RMG-Py', 'test', 'rmgpy', 'test_data',
    'testing_database', 'kinetics', 'families')


def canon(m):
    """Isomorphism-invariant canonical SMILES (explicit H, kekulized,
    sanitized) applied to BOTH sides of the comparison (rmgpu products and
    RMG-Py's recorded adjacency lists)."""
    from rdkit import Chem
    mol = m._rdkit
    if not any(a.GetSymbol() == 'H' for a in mol.GetAtoms()):
        try:
            mol = Chem.AddHs(mol)
        except Exception:  # noqa: BLE001
            pass
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:  # noqa: BLE001
        pass
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001
        pass
    return Chem.MolToSmiles(mol)


def norm(d):
    """Product-tuple -> sorted degeneracy list, rounded for float safety."""
    return {k: sorted(round(float(v), 6) for v in vs) for k, vs in d.items()}


def build_efam(loader_fam):
    """Build the enumeration Family from the loader Family (which carries
    recipe, flags, forbidden structures, template labels)."""
    return enum.Family(
        label=loader_fam.label,
        recipe=loader_fam.recipe,
        reverse_recipe=loader_fam.reverse_recipe,
        own_reverse=loader_fam.own_reverse,
        reversible=loader_fam.reversible,
        allow_charged_species=loader_fam.allow_charged_species,
        electrons=loader_fam.electrons,
        reactant_num_effective=loader_fam.num_template_reactants_effective,
        product_num_forward=loader_fam.product_num_forward,
        reverse_map=loader_fam.reverse_map,
        template_labels=[e.label for e in loader_fam.forward_template],
        forbidden=loader_fam.forbidden,
    )


def generate(fam, smiles):
    """Run rmgpu's real generation path for one case. Returns (rxns, dt)."""
    efam = build_efam(fam)
    matcher = TemplateMatcher(fam)  # loader Family is TemplateFamily-shaped
    efam.matcher = matcher
    reactants = [Molecule(smiles=s) for s in smiles]
    t0 = time.time()
    rxns = enum.generate_reactions(efam, reactants, matcher=matcher)
    return rxns, time.time() - t0, efam


def reverse_check(efam, reactants, products, matcher):
    """For an own-reverse reversible family, re-enumerate a fwd reaction's
    `products` (as reactants) and check the original `reactants` set is
    recovered among the regenerated products (resonance-relaxed, like RMG's
    model-loop reverse handling). Returns (ok, recovered_degencies)."""
    try:
        prod_mols = [p.copy(clear_labels=True) for p in products]
        rxns_rev = enum.generate_reactions(
            efam, prod_mols, matcher=matcher, delete_labels=True)
    except Exception as e:  # noqa: BLE001
        return False, '%s: %s' % (type(e).__name__, e), []
    recovers = False
    degen = []
    for r in rxns_rev:
        try:
            if enum.same_species_lists(
                    reactants, r.products, strict=False):
                recovers = True
                degen.append(float(r.degeneracy))
        except Exception:  # noqa: BLE001
            continue
    return recovers, None, degen


def main():
    results = {
        'cases_total': 0,
        'cases_ok': 0,
        'cases_mismatch': 0,
        'cases_error': 0,
        'reverse_in_scope': 0,
        'reverse_ok': 0,
        'reverse_fail': 0,
        'blocked_families': {},
        'timing_seconds': {},
        'max_timing_seconds': None,
        'timing_ok': None,
        'mismatches': [],
        'errors': [],
        'per_case': [],
    }

    with open(REFERENCE) as f:
        ref = json.load(f)
    refmap = {(c['family'], tuple(c['reactant_smiles'])): c
              for c in ref['cases']}
    results['reference_cases'] = len(ref['cases'])

    kf = KineticsFamilies().load('default')
    default_fams = {f.label: f for f in kf.families}
    results['blocked_families'] = {
        k: dict(v) for k, v in kf.blocked.items()}

    test_fams = {}
    for name in sorted({c[0] for c in CASES if c[1] == 'test'}):
        try:
            test_fams[name] = Family.from_files(name, TEST_FAM_DIR,
                                                kinetics_db=None)
        except Exception as e:  # noqa: BLE001
            results['errors'].append(
                'cannot load test family %s: %s: %s' % (name,
                                                        type(e).__name__, e))

    for fam_label, source, smiles in CASES:
        results['cases_total'] += 1
        key = (fam_label, tuple(smiles))
        rc = refmap.get(key)
        if rc is None:
            results['cases_error'] += 1
            results['errors'].append('no reference for %s %s'
                                     % (fam_label, smiles))
            continue
        lfm = (test_fams.get(fam_label) if source == 'test'
               else default_fams.get(fam_label))
        if lfm is None:
            results['cases_error'] += 1
            results['errors'].append('family not loaded: %s (%s)'
                                     % (fam_label, source))
            continue
        try:
            rxns, dt, efam = generate(lfm, smiles)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results['cases_error'] += 1
            results['errors'].append('%s %s: %s: %s'
                                     % (fam_label, smiles, type(e).__name__,
                                        e))
            continue
        results['timing_seconds']['%s|%s' % (fam_label, ','.join(smiles))] = \
            round(dt, 4)

        # 1. product sets + degeneracies, exact
        got = {}
        for r in rxns:
            prods = tuple(sorted(canon(p) for p in r.products))
            got.setdefault(prods, []).append(r.degeneracy)
        want = {}
        for rr in rc['fwd_reactions']:
            rp = [Molecule.from_adjacency_list(adj)
                  for adj in rr['products']]
            prods = tuple(sorted(canon(p) for p in rp))
            want.setdefault(prods, []).append(rr['degeneracy'])
        g, w = norm(got), norm(want)
        ok = g == w

        entry = {
            'family': fam_label,
            'source': source,
            'reactants': list(smiles),
            'n_fwd_reactions': len(rxns),
            'n_ref_reactions': rc['n_fwd_reactions'],
            'parity_ok': bool(ok),
            'timing_seconds': round(dt, 4),
            'got': {str(k): v for k, v in g.items()},
            'want': {str(k): v for k, v in w.items()},
            'reverse': [],
        }

        # 2. reverse recovery (own-reverse reversible families only)
        if lfm.own_reverse and lfm.reversible:
            for r in rxns:
                results['reverse_in_scope'] += 1
                ro, rerr, rdeg = reverse_check(
                    efam, [Molecule(smiles=s) for s in smiles],
                    list(r.products), TemplateMatcher(lfm))
                if ro:
                    results['reverse_ok'] += 1
                else:
                    results['reverse_fail'] += 1
                entry['reverse'].append({
                    'ok': bool(ro),
                    'error': rerr,
                    'degeneracy': rdeg,
                })
        # Non-own-reverse families: reverse check out of scope (None).

        if ok and all(rv['ok'] for rv in entry['reverse']):
            results['cases_ok'] += 1
        else:
            results['cases_mismatch'] += 1
            if not ok:
                results['mismatches'].append({
                    'family': fam_label,
                    'reactants': list(smiles),
                    'got': entry['got'],
                    'want': entry['want'],
                })
        results['per_case'].append(entry)

    # 3. timing
    if results['timing_seconds']:
        results['max_timing_seconds'] = max(results['timing_seconds'].values())
        results['timing_ok'] = results['max_timing_seconds'] < TIMING_SECONDS

    green = (
        results['cases_error'] == 0
        and results['cases_mismatch'] == 0
        and results['reverse_fail'] == 0
        and results['timing_ok'] is True
        and results['reference_cases'] == results['cases_total']
    )

    # ---- report ----
    lines = []
    lines.append('== JOB-05 GATE: product-enumeration parity vs RMG-Py ==')
    lines.append('cases: %d (reference %d)' % (results['cases_total'],
                                               results['reference_cases']))
    lines.append('parity exact (set + degeneracy): %d/%d'
                 % (results['cases_ok'], results['cases_total']))
    lines.append('reverse recovery (in scope): %d/%d ok'
                 % (results['reverse_ok'], results['reverse_in_scope']))
    lines.append('timing max: %s s (floor %s s): %s'
                 % (results['max_timing_seconds'], TIMING_SECONDS,
                    'OK' if results['timing_ok'] else 'FAIL'))
    lines.append('blocked families: %d %s'
                 % (len(results['blocked_families']),
                    sorted(results['blocked_families']) or '-'))
    lines.append('')
    for e in results['per_case']:
        flag = 'OK' if e['parity_ok'] else 'MISMATCH'
        lines.append('%-9s %-24s %-24s n_rxn=%d ref=%d %.3fs'
                     % (flag, e['family'], ','.join(e['reactants']),
                        e['n_fwd_reactions'], e['n_ref_reactions'],
                        e['timing_seconds']))
    if results['mismatches']:
        lines.append('')
        lines.append('MISMATCHES:')
        for m in results['mismatches']:
            lines.append(json.dumps(m, indent=2))
    if results['errors']:
        lines.append('')
        lines.append('ERRORS:')
        for err in results['errors']:
            lines.append('  ' + err)
    lines.append('')
    lines.append('GATE STATUS: %s' % ('GREEN' if green else 'RED'))
    report = '\n'.join(lines)
    print(report)

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(results, f, indent=1)
    print('\nresults -> %s' % RESULTS)
    sys.exit(0 if green else 1)


if __name__ == '__main__':
    main()
