"""
Tests for the family loader + KineticsFamilies facade (job-05/step-04):
rmgpu/core/family.py, against the recorded RMG-Py reference.

Non-circular baseline: RMG-Py's own load, recorded by
scripts/record_job05_step04_reference.py as
  gates/baselines/job05/step04_families_reference.json
which holds, for every family in the 'default' recommended set:
  - the loaded KineticsFamily's enumeration/matching fields (template
    reactant/product labels, group-tree entry count, recipe actions,
    reverse recipe, own_reverse / reversible / allow_charged_species /
    electrons / reactant_num / product_num / auto_generated / reverse_map,
    the EFFECTIVE reactant count after RMG's single-group split, and the
    effective forward product count), recorded the way RMG-Py's own
    load + family_meta semantics resolve them; and
  - match_reaction ground truth: for each of 20 (family, reaction) pairs
    (drawn from the 7 mechanism families, incl. R_Recombination's single
    group split and H_Abstraction's own-reverse relabel), the verdict of
    EVERY default family - matched + the most-specific template labels.

The parity the step asks for:
  1. all 'default' families load without error;
  2. counts (families, recipes per family, templates per family) agree
     with the reference;
  3. match_reaction on the 20 reactions attributes each to the family
     RMG-Py says matches it, with RMG-Py's labels (modulo one documented
     aromatic-representation difference, see _AROMATIC_DESCENT_EXCEPTIONS);
  4. the blocked-families list is empty for the 'default' set (the loader
     expresses every 'default' family's data; the mechanism to record a
     blocker is validated separately).
"""
import json
import os

import pytest

from rmgpu.molecule.molecule import Molecule
from rmgpu.core import template as T
from rmgpu.core.family import KineticsFamilies, Family, FamilyLoadError

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(REPO, 'gates', 'baselines', 'job05',
                   'step04_families_reference.json')

_KF = {}


def _ref():
    with open(REF) as f:
        return json.load(f)


def _kf():
    """The single loaded KineticsFamilies ('default' set) for the whole
    module (loading once keeps the match_reaction tests cheap)."""
    if 'kf' not in _KF:
        _KF['kf'] = KineticsFamilies().load('default')
    return _KF['kf']


def _reaction_for(rec):
    """A T.Reaction for a reference match-case record (reactants/products
    are RMG adjacency lists, same source the recipe engine is designed
    for - see test_template_match._reaction_by_ci)."""
    rm = [Molecule.from_adjacency_list(a) for a in rec['reactants']]
    pm = [Molecule.from_adjacency_list(a) for a in rec['products']]
    return T.Reaction(rm, pm)


# ---------------------------------------------------------------------------
# 1. All 'default' families load without error
# ---------------------------------------------------------------------------

def test_all_default_families_load():
    """Every family in RMG-Py's 'default' recommended set loads without
    error and is NOT blocked (the blocked list is empty for this set)."""
    ref = _ref()
    kf = _kf()
    assert len(ref['default']) == 51, 'default set size changed (%d)' % len(
        ref['default'])
    assert len(kf.families) == len(ref['default']), (
        'loaded %d, reference has %d' % (len(kf.families), len(ref['default'])))
    assert not kf.blocked, (
        'blocked families in the default set: %r' %
        {n: v['reason'] for n, v in kf.blocked.items()})
    for name in ref['default']:
        assert kf.get_family(name) is not None, 'family %s not loaded' % name


# ---------------------------------------------------------------------------
# 2. Counts: families, recipes per family, templates per family
# ---------------------------------------------------------------------------

def test_counts_agree_with_rmgpy():
    """Per-family counts agree with RMG-Py's recorded load: the number of
    recipe actions, the number of group-tree template entries
    (num_groups_file - the entries the loader parses from groups.py), and
    the template reactant/product labels."""
    ref = _ref()
    kf = _kf()
    bad = []
    for name in ref['default']:
        f = kf.get_family(name)
        rf = ref['families'][name]
        # recipes per family: the recipe's action count
        if len(f.recipe.actions) != len(rf['recipe']):
            bad.append((name, 'recipe_actions',
                        len(f.recipe.actions), len(rf['recipe'])))
        # templates per family: the parsed group-tree entry count
        if len(f.entries) != rf['num_groups_file']:
            bad.append((name, 'num_groups_file',
                        len(f.entries), rf['num_groups_file']))
        if f.num_groups_file != rf['num_groups_file']:
            bad.append((name, 'num_groups_file_attr',
                        f.num_groups_file, rf['num_groups_file']))
        # template reactant/product labels
        got_fwd = [e.label for e in f.forward_template]
        if got_fwd != rf['template_reactants']:
            bad.append((name, 'template_reactants', got_fwd,
                        rf['template_reactants']))
        if f.template_products != rf['template_products']:
            bad.append((name, 'template_products', f.template_products,
                        rf['template_products']))
    assert not bad, 'count mismatches vs RMG-Py: %r' % (bad,)


def test_reverse_family_bookkeeping():
    """The reverse-family bookkeeping (reverse / reversible / own_reverse /
    reverse_map / the effective reactant + product counts) matches the
    reference for every default family - this is what job 06 uses to know a
    reaction is reversible."""
    ref = _ref()
    kf = _kf()
    bad = []
    for name in ref['default']:
        f = kf.get_family(name)
        rf = ref['families'][name]
        checks = {
            'own_reverse': (f.own_reverse, rf['own_reverse']),
            'reversible': (f.reversible, rf['reversible']),
            'reverse_map': (f.reverse_map, rf['reverse_map']),
            'allow_charged_species': (f.allow_charged_species,
                                      rf['allow_charged_species']),
            'electrons': (f.electrons, rf['electrons']),
            'reactant_num_stored': (f.reactant_num_stored, rf['reactant_num']),
            'num_template_reactants_effective': (
                f.num_template_reactants_effective,
                rf['num_template_reactants_effective']),
            'product_num_forward': (f.product_num_forward,
                                    rf['effective_product_num_forward']),
        }
        for k, (got, want) in checks.items():
            if got != want:
                bad.append((name, k, got, want))
    assert not bad, 'reverse-bookkeeping mismatches vs RMG-Py: %r' % (bad,)


# ---------------------------------------------------------------------------
# 3. match_reaction on the 20 reactions agrees with RMG-Py
# ---------------------------------------------------------------------------

# The two benzene substrates: rmgpu stores molecules Kekulized (Cd carbons,
# single/double bonds), RMG-Py stores them aromatic (Cb carbons, 1.5
# bonds). The match VERDICT and the reactant->template labeling agree with
# RMG-Py for both; only the most-specific DESCENT leaf differs (Cd-subtree
# vs Cb-subtree) - a representation difference, not a matching bug. Same
# documented exception as step-03 (test_template_match.py).
_AROMATIC_DESCENT_EXCEPTIONS = {
    ('H_Abstraction', 4, 0),
    ('R_Addition_MultipleBond', 19, 0),
}


def _case_key(rec):
    return '%s|%d|%d' % (rec['own'], rec['ci'], rec['ri'])


def test_match_reaction_family_agreement():
    """For each of the 20 reactions, match_reaction attributes it to the
    family RMG-Py says matches it (the reference verdicts record exactly
    one matching family per reaction)."""
    ref = _ref()
    kf = _kf()
    n = 0
    bad = []
    for rec in ref['reactions']:
        key = _case_key(rec)
        row = ref['verdicts'][key]
        ref_matchers = [name for name, v in row.items() if v['matched']]
        assert len(ref_matchers) == 1, (
            'reference must record exactly one matcher for %s, got %r'
            % (key, ref_matchers))
        rxn = _reaction_for(rec)
        fam, labels = kf.match_reaction(rxn)
        got = fam.label if fam is not None else None
        if got == ref_matchers[0]:
            n += 1
        else:
            bad.append((key, got, ref_matchers))
    assert not bad, 'match_reaction family disagreements: %r' % (bad,)
    assert n == len(ref['reactions']) == 20


def test_match_reaction_label_agreement():
    """For each of the 20 reactions, match_reaction's descended template
    labels equal RMG-Py's recorded labels, except the two documented
    aromatic-representation cases (verdict + labeling still agree there)."""
    ref = _ref()
    kf = _kf()
    agree = 0
    excepted = 0
    bad = []
    for rec in ref['reactions']:
        key = _case_key(rec)
        row = ref['verdicts'][key]
        ref_matchers = [name for name, v in row.items() if v['matched']]
        ref_labels = row[ref_matchers[0]]['template']
        rxn = _reaction_for(rec)
        fam, labels = kf.match_reaction(rxn)
        assert fam is not None, 'own-family match returned None: %s' % key
        if labels == ref_labels:
            agree += 1
        elif (rec['own'], rec['ci'], rec['ri']) in _AROMATIC_DESCENT_EXCEPTIONS:
            excepted += 1
        else:
            bad.append((key, labels, ref_labels))
    assert not bad, 'unexpected template-label disagreements: %r' % (bad,)
    assert agree + excepted == 20
    assert excepted == len(_AROMATIC_DESCENT_EXCEPTIONS)


# ---------------------------------------------------------------------------
# 4. The blocked-families mechanism (the list is load-bearing for job 06)
# ---------------------------------------------------------------------------

def test_blocked_mechanism_records_construct_and_note():
    """The blocked-families list is empty for the 'default' set, but the
    mechanism that populates it records, for each blocker, the reason, the
    exact construct that failed, and a note for the gate. Verified with a
    synthetic blocker (a family with no groups.py) in a temp dir."""
    import tempfile
    ref = _ref()
    kf = _kf()
    assert not kf.blocked, (
        'expected an empty blocked list for the default set, got %r'
        % {n: v['reason'] for n, v in kf.blocked.items()})
    with tempfile.TemporaryDirectory() as d:
        bad = KineticsFamilies(fam_dir=d, kinetics_db=None).load(
            ['DoesNotExist'])
        assert 'DoesNotExist' in bad.blocked
        info = bad.blocked['DoesNotExist']
        assert 'reason' in info and 'construct' in info and 'note' in info
        assert info['note']
        assert 'DoesNotExist' not in bad._families


def test_family_load_error_on_missing_file():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(FamilyLoadError):
            Family.from_files('NoSuchFamily', d, None)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
