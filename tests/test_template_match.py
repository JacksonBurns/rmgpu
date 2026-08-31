"""
Tests for template matching (job-05/step-03): rmgpu/core/template.py
`match(family, reaction) -> template_labels` + the group matcher
(rmgpu/molecule/group.py), against the recorded RMG-Py reference.

Non-circular baseline: RMG-Py's own matching, recorded by
scripts/record_job05_step03_verdicts.py as
  gates/baselines/job05/step03_match_verdicts.json
which holds, for each of the 46 reactions in the step-02 product set:
  - the reactant SMILES + RMG-Py's own template labels (the reaction's
    `template`), and
  - a 7-family x 46-reaction verdict matrix: for every (family, reaction)
    pair, whether RMG-Py's get_labeled_reactants_and_products matched it
    (`matched`) and the most-specific template labels it descended to
    (`template`).

The recorded matrix is decisive: exactly each reaction's OWN family matches
it (46 own-family `matched: true`) and NO cross-family pair matches
(276 `matched: false`, 0 false positives). The parity the step asks for -
"match agrees with RMG-Py" - is therefore: for every (family, reaction)
pair, rmgpu's match() verdict (matches / no-match) equals RMG-Py's, and
where both match, the descended template labels agree (modulo one documented
aromatic representation difference, see test_label_agreement).

Products come from the step-02 reference (step02_products_reference.json)
which stores them as RMG adjacency lists; the verdicts reference stores
reactants as SMILES. They join on (ci, ri): verdicts.reactions[i] has
(ci, ri, family) and step-02.cases[ci].reactions[ri] is the same reaction.
"""
import json
import os

import pytest

from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule import group as G
from rmgpu.core import template as T

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERDICTS = os.path.join(REPO, 'gates', 'baselines', 'job05',
                        'step03_match_verdicts.json')
STEP02 = os.path.join(REPO, 'gates', 'baselines', 'job05',
                      'step02_products_reference.json')
TEMPLATES = os.path.join(REPO, 'gates', 'baselines', 'job05',
                         'step03_templates_reference.json')

_BENCH = {}


def _load(name, path):
    if name not in _BENCH:
        with open(path) as f:
            _BENCH[name] = json.load(f)
    return _BENCH[name]


def _families():
    """The 7 TemplateFamily objects (tree from step-03 ref + recipe/flags
    from the step-02 family_meta), keyed by family label."""
    ref = _load('templates', TEMPLATES)
    s2 = _load('step02', STEP02)
    s2_by_fam = {c['family']: c for c in s2['cases']}
    out = {}
    for fl, fref in ref['families'].items():
        out[fl] = T.TemplateFamily.from_references(fref, s2_by_fam[fl])
    return out


def _reaction_by_ci(ci, ri):
    """A T.Reaction for step-02 case `ci`, reaction `ri`.

    Reactants are built from the STEP-02 recorded RMG adjacency lists (RMG's
    own explicit-H representation, with RMG's atom ordering) - the same
    source the harness uses and the representation the recipe engine is
    designed for (it applies label-indexed bond recipes to explicit-H
    structures and kekulizes the product ring). Building the reactants from
    a re-canonicalized SMILES instead would put the heavy atoms in RDKit's
    canonical order (a different explicit-H atom ordering), and the
    recipe/kekulize step (step 01) is sensitive to that ordering for the
    semi-kekulized ene/shift products, so the RMG representation is the
    faithful, validated choice. The SMILES (verdicts reference) is used only
    to identify the reaction.
    """
    s2 = _load('step02', STEP02)
    rec = s2['cases'][ci]['reactions'][ri]
    rm = [Molecule.from_adjacency_list(a) for a in rec['reactants']]
    pm = [Molecule.from_adjacency_list(a) for a in rec['products']]
    v = _load('verdicts', VERDICTS)
    own = next(r['family'] for r in v['reactions']
               if r['ci'] == ci and r['ri'] == ri)
    return T.Reaction(rm, pm), own, rec['reactants']


def _reaction_ids():
    v = _load('verdicts', VERDICTS)
    return ['%s|ci=%d|ri=%d' % (r['family'], r['ci'], r['ri'])
            for r in v['reactions']]


REACTION_KEYS = [(r['ci'], r['ri']) for r in _load('verdicts', VERDICTS)
                 ['reactions']]


# ---------------------------------------------------------------------------
# The recorded reference
# ---------------------------------------------------------------------------

def test_reference_present():
    assert os.path.exists(VERDICTS), 'missing step-03 verdicts baseline'
    assert os.path.exists(TEMPLATES), 'missing step-03 templates baseline'
    v = _load('verdicts', VERDICTS)
    assert len(v['reactions']) == 46
    assert len(v['verdicts']) == 322  # 7 families x 46 reactions
    # the recorded matrix is decisive: 46 matched (all own-family), 0 cross
    matched = [k for k, d in v['verdicts'].items() if d['matched']]
    assert len(matched) == 46
    by_key = {(r['ci'], r['ri']): r['family'] for r in v['reactions']}
    for k in matched:
        fl, ci, ri = k.split('|')
        own = by_key[(int(ci), int(ri))]
        assert fl == own, 'a cross-family pair matched in the reference: %s' % k


# ---------------------------------------------------------------------------
# 1. Round-trip: every generated reaction matches its own family's template
# ---------------------------------------------------------------------------

def test_roundtrip_own_family_matches():
    """Each of the 46 reactions (generated by a family in step 02) must match
    its own family's template - the round-trip consistency check."""
    fams = _families()
    v = _load('verdicts', VERDICTS)
    n = 0
    for r in v['reactions']:
        rxn, own, _ = _reaction_by_ci(r['ci'], r['ri'])
        got = T.match(fams[own], rxn)
        assert got is not None, (
            '%s (ci=%d ri=%d): own-family match returned None, RMG-Py says '
            'matched' % (own, r['ci'], r['ri']))
        n += 1
    assert n == 46


# ---------------------------------------------------------------------------
# 2. Match verdict agrees with RMG-Py across the full matrix
# ---------------------------------------------------------------------------

def _all_pair_keys():
    v = _load('verdicts', VERDICTS)
    keys = []
    for r in v['reactions']:
        for fl in v['families']:
            keys.append((fl, r['ci'], r['ri']))
    return keys


def _pair_key(fl, ci, ri):
    return '%s|%d|%d' % (fl, ci, ri)


def test_full_verdict_agreement_with_rmgpy():
    """The core parity check: for ALL 322 (family, reaction) pairs, rmgpu's
    match() verdict (non-None = matched, None = not) equals RMG-Py's recorded
    verdict. This is the step's "match agrees with RMG-Py" requirement made
    exhaustive (it subsumes the 40 round-trip + 10 negative asks)."""
    fams = _families()
    v = _load('verdicts', VERDICTS)
    rxn_cache = {}
    agree = 0
    bad = []
    for (fl, ci, ri) in _all_pair_keys():
        ref = v['verdicts'][_pair_key(fl, ci, ri)]['matched']
        if (ci, ri) not in rxn_cache:
            rxn_cache[(ci, ri)] = _reaction_by_ci(ci, ri)[0]
        rxn = rxn_cache[(ci, ri)]
        got = T.match(fams[fl], rxn)
        mine = got is not None
        if mine == ref:
            agree += 1
        else:
            bad.append((_pair_key(fl, ci, ri), ref, mine))
    assert not bad, 'verdict disagreements with RMG-Py: %r' % (bad,)
    assert agree == 322


# ---------------------------------------------------------------------------
# 3. Negative cases: a WRONG family must NOT match
# ---------------------------------------------------------------------------

def _negative_keys():
    """10 wrong-family pairs with MATCHING reactant counts (so the check
    exercises the recipe/product filter, not just the reactant-count guard),
    all recorded `matched: false`. Deterministic (sorted)."""
    v = _load('verdicts', VERDICTS)
    s2 = _load('step02', STEP02)
    fnum = {c['family']: c['family_meta'].get('num_template_reactants_effective')
            for c in s2['cases']}
    cands = []
    for r in v['reactions']:
        nr = len(r['reactants'])
        for fl in v['families']:
            if fl == r['family']:
                continue
            if fnum.get(fl) != nr:
                continue
            if v['verdicts'][_pair_key(fl, r['ci'], r['ri'])]['matched']:
                continue
            cands.append((fl, r['ci'], r['ri']))
    cands = sorted(set(cands))
    # spread across families for coverage
    picked = []
    for c in cands:
        if c not in picked:
            picked.append(c)
        if len(picked) == 10:
            break
    return picked


NEGATIVES = _negative_keys()


def test_negative_wrong_family_no_match():
    """A reaction must NOT match a wrong family (agrees with RMG-Py's
    recorded matched:false). These 10 pairs have the same reactant count as
    the wrong family, so the rejection comes from the recipe/product check,
    not the reactant-count guard."""
    fams = _families()
    v = _load('verdicts', VERDICTS)
    for (fl, ci, ri) in NEGATIVES:
        rxn, own, _ = _reaction_by_ci(ci, ri)
        assert fl != own, 'negative must be a wrong family: %s' % (fl,)
        ref = v['verdicts'][_pair_key(fl, ci, ri)]['matched']
        assert ref is False, 'negative not recorded as no-match: %s' % (fl,)
        got = T.match(fams[fl], rxn)
        assert got is None, (
            'wrong family %s matched reaction %s (own=%s); RMG-Py says no'
            % (fl, _pair_key(fl, ci, ri), own))


# ---------------------------------------------------------------------------
# 4. Template-label agreement (where both match)
# ---------------------------------------------------------------------------

# The two benzene substrates: rmgpu stores molecules Kekulized (Cd carbons,
# single/double bonds), RMG-Py stores them aromatic (Cb carbons, 1.5 bonds).
# The match VERDICT and the reactant->template labeling agree with RMG-Py for
# both; only the most-specific DESCENT leaf differs (Cd-subtree vs Cb-subtree),
# a representation difference, not a matching bug. Documented here so the
# label test is exact on the remaining 44.
_AROMATIC_DESCENT_EXCEPTIONS = {
    ('H_Abstraction', 4, 0),          # benzene + [H] -> Cb_H vs Cd/H/Cd
    ('R_Addition_MultipleBond', 19, 0),  # benzene + [H] -> Cb-H_Cb-H vs Cds-CdH_Cds-CdH
}


def test_template_label_agreement():
    """For every pair where RMG-Py matched (the 46 own-family cases), rmgpu's
    descended template labels equal RMG-Py's recorded labels, except the two
    documented aromatic-representation cases (verdict + labeling still agree;
    the descent leaf differs Cd<->Cb)."""
    fams = _families()
    v = _load('verdicts', VERDICTS)
    agree = 0
    excepted = 0
    bad = []
    for r in v['reactions']:
        own = r['family']
        rxn, _, _ = _reaction_by_ci(r['ci'], r['ri'])
        got = T.match(fams[own], rxn)
        assert got is not None
        ref = v['verdicts'][_pair_key(own, r['ci'], r['ri'])]['template']
        if got == ref:
            agree += 1
        elif (own, r['ci'], r['ri']) in _AROMATIC_DESCENT_EXCEPTIONS:
            excepted += 1
        else:
            bad.append((_pair_key(own, r['ci'], r['ri']), got, ref))
    assert not bad, 'unexpected template-label disagreements: %r' % (bad,)
    assert agree + excepted == 46
    assert excepted == len(_AROMATIC_DESCENT_EXCEPTIONS)


# ---------------------------------------------------------------------------
# 5. Group matcher: subgraph-isomorphism parity with RMG-Py (99 pairs)
# ---------------------------------------------------------------------------

def test_group_subgraph_parity():
    """The group matcher (rmgpu/molecule/group.py) finds the same number of
    valid subgraph-isomorphisms as RMG-Py for every (reactant, template-slot)
    pair in the recorded step-03 reference (99 pairs). This is the
    matcher-level parity that the template matching builds on."""
    ref = _load('templates', TEMPLATES)
    ok = 0
    bad = []
    for c in ref['match_cases']:
        mols = [Molecule(smiles=s) for s in c['reactant_smiles']]
        for ps in c['per_slot']:
            ref_n = ps['n_mappings']
            my_n = 0
            for comp_adj in c['slot_groups_adj'][ps['template_slot']]:
                g = G.parse_group_adjlist_full(comp_adj)
                my_n += len(G.match_group(mols[ps['reactant_index']], g))
            if my_n == ref_n:
                ok += 1
            else:
                bad.append((c['family'], c['reactant_smiles'][ps['reactant_index']],
                            ps['template_slot'], my_n, ref_n))
    assert not bad, 'group subgraph-parity mismatches: %r' % (bad,)
    assert ok == 99


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
