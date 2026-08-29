"""
job-05/step-01 checks: the ReactionRecipe engine (rmgpu/core/recipe.py).

Ground truth: gates/baselines/job05/step01_apply_recipe_reference.json,
recorded by scripts/record_job05_step01_reference.py running RMG-Py's
KineticsFamily.apply_recipe (rmg_env, RMG-Py's own testing_database
families - attribution: MIT license, RMG-Py repo). The recorded cases
carry the input adjacency lists verbatim plus each family's recipe and
metadata (own_reverse, reverse_map, effective product counts, electrons),
so this test file has no rmgpy dependency.

Each case checks, in RMG's product order:
- the piece count (product_num),
- the per-piece atom count and net charge (RMG update() applied),
- the per-label structural fingerprint (element, radical electrons,
  formal charge, sorted neighbor bond orders) - order-independent.

Plus unit tests for the recipe object itself (get_reverse) and the
validity rules (ActionError paths).
"""
import json
import os

import pytest

from rmgpu.core.recipe import (
    ActionError,
    ReactionRecipe,
    apply_recipe,
    label_fingerprint,
    label_atoms,
    clear_labeled_atoms,
)
from rmgpu.molecule.molecule import Molecule

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join(REPO, "gates", "baselines", "job05",
                         "step01_apply_recipe_reference.json")

# RMG-Py testing_database family label -> lowercase engine label (the
# hardcoded relabel keys use the family label lowercased).
FAMILY_LABELS = {
    "H_Abstraction": "h_abstraction",
    "R_Addition_MultipleBond_benzene": "r_addition_multiplebond",
    "intra_H_migration": "intra_h_migration",
    "Intra_ene_reaction": "intra_ene_reaction",
    "6_membered_central_C-C_shift": "6_membered_central_c-c_shift",
    "1,2_shiftC": "1,2_shiftc",
    "Intra_R_Add_Exo_scission": "intra_r_add_exo_scission",
    "intra_substitutionS_isomerization": "intra_substitutions_isomerization",
    "R_Addition_COm": "r_addition_com",
}

SURFACE_CASES = (
    "Surface_Dissociation_Charge_Separation_fwd",
    "Surface_Dissociation_Charge_Separation_rev",
)


@pytest.fixture(scope="module")
def reference():
    with open(REFERENCE) as f:
        return json.load(f)


def _run_case(reference, name, family_label=None):
    """Run one recorded case through the rmgpu engine and return the
    product Molecules (None if the engine reported no match)."""
    case = reference[name]
    mols = [Molecule.from_adjacency_list(t) for t in case["reactants"]]
    recipe = ReactionRecipe.from_data(case["recipe"])
    fwd = case["forward"]
    pnum = (case["effective_product_num_forward"] if fwd
            else case["effective_product_num_reverse"])
    return apply_recipe(
        mols, recipe,
        family_label=family_label if family_label is not None
        else FAMILY_LABELS.get(name, name.lower().replace("-", "_")),
        forward=fwd,
        own_reverse=case["own_reverse"],
        reverse_map=case.get("reverse_map"),
        product_num=pnum,
        electrons=case["electrons"],
        relabel_atoms=True,
    )


# ---------------------------------------------------------------------------
# Case-by-case parity with the RMG-Py recorded products
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [
        "H_Abstraction",
        "R_Addition_MultipleBond_benzene",
        "intra_H_migration",
        "Intra_ene_reaction",
        "6_membered_central_C-C_shift",
        "1,2_shiftC",
        "Intra_R_Add_Exo_scission",
        "intra_substitutionS_isomerization",
        "R_Addition_COm",
    ],
)
def test_apply_recipe_case(reference, name):
    case = reference[name]
    prods = _run_case(reference, name)
    assert prods is not None, "engine reported no product (template mismatch)"
    assert len(prods) == len(case["products"]), \
        "piece count mismatch: got {}, expected {}".format(
            len(prods), len(case["products"]))
    for i, (p, exp) in enumerate(zip(prods, case["products"])):
        got_n = len(p._rdkit.GetAtoms())
        assert got_n == exp["num_atoms"], \
            "piece {}: atom count {} != {}".format(i, got_n, exp["num_atoms"])
        got_charge = sum(a.GetFormalCharge()
                         for a in p._rdkit.GetAtoms())
        assert got_charge == exp["net_charge"], \
            "piece {}: net charge {} != {}".format(
                i, got_charge, exp["net_charge"])
        got_fp = label_fingerprint(p)
        assert got_fp == exp["label_fingerprint"], \
            "piece {}: label fingerprint mismatch:\n  got      {}\n  expected {}".format(
                i, json.dumps(got_fp, sort_keys=True),
                json.dumps(exp["label_fingerprint"], sort_keys=True))


def test_h_abstraction_product_order(reference):
    """RMG orders two products with the '*1' piece first: after the
    h_abstraction relabel, '*1' is on the abstracted H (the H2 piece), so
    H2 comes first and the acetyl radical second."""
    case = reference["H_Abstraction"]
    prods = _run_case(reference, "H_Abstraction")
    assert len(prods) == 2
    # piece 0 = H2 (labels *1, *2); piece 1 = acetyl (label *3)
    assert label_fingerprint(prods[0]) == case["products"][0]["label_fingerprint"]
    assert label_fingerprint(prods[1]) == case["products"][1]["label_fingerprint"]


def test_self_reverse_relabel_chain_swap(reference):
    """intra_H_migration relabels the chain ends (*1<->*2) and reverses
    the chain (*4<->*5, *6<->*7) - the recorded fingerprints carry the
    relabeled placement (the radical sits on the label the H left, which
    after the swap is '*1')."""
    case = reference["intra_H_migration"]
    prods = _run_case(reference, "intra_H_migration")
    assert len(prods) == 1
    fp = label_fingerprint(prods[0])
    # the radical ends up on the atom carrying label *1 after relabel
    assert fp["*1"]["radical"] == 1
    assert fp == case["products"][0]["label_fingerprint"]


def test_r_addition_com_pair_charge_bookkeeping(reference):
    """R_Addition_COm: the LOSE_PAIR/GAIN_PAIR actions recompute the
    formal charges (RMG update_charge coupling); the final product piece
    carries the net charge 0 with O(*3) lp 2 and C(*1) radical 1."""
    case = reference["R_Addition_COm"]
    prods = _run_case(reference, "R_Addition_COm")
    assert len(prods) == 1
    fp = label_fingerprint(prods[0])
    assert fp["*3"] == {"element": "O", "radical": 0, "charge": 0,
                        "bonds": [2.0]}
    assert fp["*1"]["radical"] == 1


def test_surface_cases_recorded_but_not_supported(reference):
    """The Surface_Dissociation_Charge_Separation cases involve 'X'
    surface sites - recorded for job-12 reference only; the gas-phase
    engine cannot build them (element 'X' is not a real element)."""
    for name in SURFACE_CASES:
        case = reference[name]
        with pytest.raises(Exception):
            _run_case(reference, name)


# ---------------------------------------------------------------------------
# ReactionRecipe object
# ---------------------------------------------------------------------------

def test_recipe_from_data_actions():
    r = ReactionRecipe.from_data(
        [["BREAK_BOND", "*1", 1, "*2"],
         ["FORM_BOND", "*2", 1, "*3"],
         ["GAIN_RADICAL", "*1", "1"],
         ["LOSE_RADICAL", "*3", "1"]])
    assert r.actions == [
        ["BREAK_BOND", "*1", 1, "*2"],
        ["FORM_BOND", "*2", 1, "*3"],
        ["GAIN_RADICAL", "*1", "1"],
        ["LOSE_RADICAL", "*3", "1"],
    ]


def test_recipe_from_data_json_string():
    """rmgdb stores the recipe as a JSON string with an 'actions' key."""
    r = ReactionRecipe.from_data(
        json.dumps({"actions": [["FORM_BOND", "*1", 1, "*2"]]}))
    assert r.actions == [["FORM_BOND", "*1", 1, "*2"]]


def test_recipe_unknown_action_raises():
    with pytest.raises(ActionError):
        ReactionRecipe.from_data([["TELEPORT", "*1", 1, "*2"]])


def test_recipe_get_reverse(reference):
    """get_reverse must match RMG-Py's recorded reverse recipe exactly
    (H_Abstraction)."""
    fwd = ReactionRecipe.from_data(reference["H_Abstraction"]["recipe"])
    rev = fwd.get_reverse()
    assert [list(a) for a in rev.actions] == reference["H_Abstraction_recipe_rev"]


def test_recipe_get_reverse_bond_swap():
    """FORM/BREAK swap, CHANGE sign flip, radical/charge/pair swap."""
    r = ReactionRecipe.from_data(
        [["CHANGE_BOND", "*1", 1, "*2"],
         ["FORM_BOND", "*1", 1, "*3"],
         ["BREAK_BOND", "*2", 1, "*3"],
         ["GAIN_RADICAL", "*1", "1"],
         ["LOSE_CHARGE", "*2", "1"],
         ["GAIN_PAIR", "*3", "1"]])
    rev = r.get_reverse()
    assert rev.actions == [
        ["LOSE_PAIR", "*3", "1"],
        ["GAIN_CHARGE", "*2", "1"],
        ["LOSE_RADICAL", "*1", "1"],
        ["FORM_BOND", "*2", 1, "*3"],
        ["BREAK_BOND", "*1", 1, "*3"],
        ["CHANGE_BOND", "*1", "-1", "*2"],
    ]


# ---------------------------------------------------------------------------
# Validity rules (ActionError paths)
# ---------------------------------------------------------------------------

def _labeled_ethane_h():
    """Ethane C*1H*2H3 + H*3 radical as labeled rmgpu Molecules."""
    donor = Molecule.from_adjacency_list(
        "1 *1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}\n"
        "2 *2 H u0 p0 c0 {1,S}\n"
        "3    H u0 p0 c0 {1,S}\n"
        "4    H u0 p0 c0 {1,S}\n"
        "5    H u0 p0 c0 {1,S}\n")
    h = Molecule.from_adjacency_list("1 *3 H u1 p0 c0\n")
    return donor, h


def test_change_bond_nonexistent_bond_raises():
    donor, h = _labeled_ethane_h()
    recipe = ReactionRecipe.from_data([["CHANGE_BOND", "*2", 1, "*3"]])
    with pytest.raises(ActionError):
        apply_recipe([donor, h], recipe, product_num=2)


def test_form_bond_existing_bond_raises():
    donor, h = _labeled_ethane_h()
    # C*1-H*2 already bonded
    recipe = ReactionRecipe.from_data([["FORM_BOND", "*1", 1, "*2"]])
    with pytest.raises(ActionError):
        apply_recipe([donor, h], recipe, product_num=2)


def test_break_bond_nonexistent_bond_raises():
    donor, h = _labeled_ethane_h()
    # H*2 and H*3 are not bonded to each other
    recipe = ReactionRecipe.from_data([["BREAK_BOND", "*2", 1, "*3"]])
    with pytest.raises(ActionError):
        apply_recipe([donor, h], recipe, product_num=2)


def test_lose_radical_without_radical_raises():
    """LOSE_RADICAL on an atom with 0 radical electrons -> ActionError
    (RMG decrement_radical: count would go negative)."""
    m = Molecule.from_adjacency_list("1 *1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}\n"
                                     "2    H u0 p0 c0 {1,S}\n"
                                     "3    H u0 p0 c0 {1,S}\n"
                                     "4    H u0 p0 c0 {1,S}\n"
                                     "5    H u0 p0 c0 {1,S}\n")
    recipe = ReactionRecipe.from_data([["LOSE_RADICAL", "*1", "1"]])
    with pytest.raises(ActionError):
        apply_recipe([m], recipe, product_num=1)


def test_lose_pair_without_pair_raises():
    """LOSE_PAIR on an atom with 0 lone pairs -> ActionError."""
    m = Molecule.from_adjacency_list("1 *1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}\n"
                                     "2    H u0 p0 c0 {1,S}\n"
                                     "3    H u0 p0 c0 {1,S}\n"
                                     "4    H u0 p0 c0 {1,S}\n"
                                     "5    H u0 p0 c0 {1,S}\n")
    recipe = ReactionRecipe.from_data([["LOSE_PAIR", "*1", "1"]])
    with pytest.raises(ActionError):
        apply_recipe([m], recipe, product_num=1)


def test_missing_label_raises():
    m = Molecule.from_adjacency_list("1    C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}\n"
                                     "2    H u0 p0 c0 {1,S}\n"
                                     "3    H u0 p0 c0 {1,S}\n"
                                     "4    H u0 p0 c0 {1,S}\n"
                                     "5    H u0 p0 c0 {1,S}\n")
    recipe = ReactionRecipe.from_data([["BREAK_BOND", "*1", 1, "*2"]])
    with pytest.raises((ActionError, ValueError)):
        apply_recipe([m], recipe, product_num=1)


# ---------------------------------------------------------------------------
# Net-charge check (template not a match -> None)
# ---------------------------------------------------------------------------

def test_net_charge_mismatch_returns_none():
    """A reactant with a net charge the products cannot reproduce makes
    apply_recipe return None (RMG: not a template match). The reactant
    carbon carries an explicit net +1 that RMG's update() recompute
    (neutral formula) erases from the products, so reactants (+1) !=
    products (0) -> None."""
    m = Molecule.from_adjacency_list("1 *1 C u0 p0 c+1 {2,S} {3,S} {4,S} {5,S}\n"
                                     "2    H u0 p0 c0 {1,S}\n"
                                     "3    H u0 p0 c0 {1,S}\n"
                                     "4    H u0 p0 c0 {1,S}\n"
                                     "5    H u0 p0 c0 {1,S}\n")
    assert m.get_charge() == 1
    recipe = ReactionRecipe.from_data([["GAIN_RADICAL", "*1", "1"],
                                       ["LOSE_RADICAL", "*1", "1"]])
    assert apply_recipe([m], recipe, product_num=1) is None


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def test_label_atoms_and_fingerprint():
    m = Molecule.from_adjacency_list("1    C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}\n"
                                     "2    H u0 p0 c0 {1,S}\n"
                                     "3    H u0 p0 c0 {1,S}\n"
                                     "4    H u0 p0 c0 {1,S}\n"
                                     "5    H u0 p0 c0 {1,S}\n")
    label_atoms([m], [{0: "*1"}])
    fp = label_fingerprint(m)
    assert "*1" in fp
    assert fp["*1"]["element"] == "C"
    clear_labeled_atoms([m])
    assert label_fingerprint(m) == {}


def test_benzene_product_structure(reference):
    """R_Addition_MultipleBond on benzene: the product is the
    cyclohexadienyl radical (C6H7, 1 radical, cyclic). The canonical
    SMILES can differ from RMG's exact kekulization, so the structure is
    verified by formula + radical count + ring presence (the label
    fingerprint is already checked in test_apply_recipe_case)."""
    prods = _run_case(reference, "R_Addition_MultipleBond_benzene")
    assert prods is not None
    p = prods[0]
    assert p.get_formula() == "C6H7"
    assert p.get_radical_count() == 1
    assert p.is_cyclic()
