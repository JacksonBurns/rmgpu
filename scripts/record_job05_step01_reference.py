#!/usr/bin/env python3
"""
job-05/step-01 reference capture.

Runs RMG-Py's KineticsFamily.apply_recipe (rmg_env) on the exact fixture
structures from RMG-Py's test/rmgpy/data/kinetics/familyTest.py (attribution:
MIT license, RMG-Py repo) and records the ground-truth products (SMILES,
net charge, per-label structural fingerprints) to
gates/baselines/job05/step01_apply_recipe_reference.json.

This is the non-circular reference for rmgpu/core/recipe.py: it uses only
rmgpy + the RMG-Py testing database, never rmgpu code.

Each case in the JSON carries the input adjacency lists verbatim so the
rmgpu test can reconstruct the labeled reactants and the expected products
without any rmgpy dependency.

Run with: /home/jackson/miniforge3/envs/rmg_env/bin/python scripts/record_job05_step01_reference.py
"""
import json
import os

from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.molecule import Molecule
from rmgpy.reaction import Reaction

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- fixtures
# Adjacency lists copied verbatim from RMG-Py test/rmgpy/data/kinetics/familyTest.py
ADJ_H_ABSTRACTION_DONOR = """
1 *1 C u0 p0 c0 {2,S} {4,S} {5,S} {6,S}
2    C u0 p0 c0 {1,S} {3,D} {7,S}
3    C u0 p0 c0 {2,D} {8,S} {9,S}
4 *2 H u0 p0 c0 {1,S}
5    H u0 p0 c0 {1,S}
6    H u0 p0 c0 {1,S}
7    H u0 p0 c0 {2,S}
8    H u0 p0 c0 {3,S}
9    H u0 p0 c0 {3,S}
"""
ADJ_H_RADICAL = "1 *3 H u1 p0 c0"

ADJ_BENZENE = """
1  *1 C u0 p0 c0 {2,B} {6,B} {7,S}
2  *2 C u0 p0 c0 {1,B} {3,B} {8,S}
3     C u0 p0 c0 {2,B} {4,B} {9,S}
4     C u0 p0 c0 {3,B} {5,B} {10,S}
5     C u0 p0 c0 {4,B} {6,B} {11,S}
6     C u0 p0 c0 {1,B} {5,B} {12,S}
7     H u0 p0 c0 {1,S}
8     H u0 p0 c0 {2,S}
9     H u0 p0 c0 {3,S}
10    H u0 p0 c0 {4,S}
11    H u0 p0 c0 {5,S}
12    H u0 p0 c0 {6,S}
"""

ADJ_INTRA_H_MIGRATION = """
multiplicity 2
1  *2 C u0 p0 c0 {3,S} {11,S} {12,S} {13,S}
2  *4 C u0 p0 c0 {4,S} {5,S} {6,D}
3  *5 C u0 p0 c0 {1,S} {7,D} {14,S}
4  *1 C u1 p0 c0 {2,S} {8,S} {15,S}
5     C u0 p0 c0 {2,S} {10,D} {17,S}
6  *6 C u0 p0 c0 {2,D} {7,S} {19,S}
7  *7 C u0 p0 c0 {3,D} {6,S} {21,S}
8     C u0 p0 c0 {4,S} {9,D} {16,S}
9     C u0 p0 c0 {8,D} {10,S} {20,S}
10    C u0 p0 c0 {5,D} {9,S} {18,S}
11 *3 H u0 p0 c0 {1,S}
12    H u0 p0 c0 {1,S}
13    H u0 p0 c0 {1,S}
14    H u0 p0 c0 {3,S}
15    H u0 p0 c0 {4,S}
16    H u0 p0 c0 {8,S}
17    H u0 p0 c0 {5,S}
18    H u0 p0 c0 {10,S}
19    H u0 p0 c0 {6,S}
20    H u0 p0 c0 {9,S}
21    H u0 p0 c0 {7,S}
"""

ADJ_INTRA_ENE = """
1  *1 C u0 p0 c0 {2,S} {3,S} {4,S} {10,S}
2  *5 C u0 p0 c0 {1,S} {5,D} {6,S}
3  *2 C u0 p0 c0 {1,S} {7,D} {11,S}
4     C u0 p0 c0 {1,S} {8,D} {12,S}
5  *4 C u0 p0 c0 {2,D} {7,S} {13,S}
6     C u0 p0 c0 {2,S} {9,D} {15,S}
7  *3 C u0 p0 c0 {3,D} {5,S} {14,S}
8     C u0 p0 c0 {4,D} {9,S} {17,S}
9     C u0 p0 c0 {6,D} {8,S} {16,S}
10 *6 H u0 p0 c0 {1,S}
11    H u0 p0 c0 {3,S}
12    H u0 p0 c0 {4,S}
13    H u0 p0 c0 {5,S}
14    H u0 p0 c0 {7,S}
15    H u0 p0 c0 {6,S}
16    H u0 p0 c0 {9,S}
17    H u0 p0 c0 {8,S}
"""

ADJ_6MEM_SHIFT = """
1  *3 C u0 p0 c0 {2,S} {3,S} {7,S} {8,S}
2  *4 C u0 p0 c0 {1,S} {4,S} {9,S} {10,S}
3  *2 C u0 p0 c0 {1,S} {5,T}
4  *5 C u0 p0 c0 {2,S} {6,T}
5  *1 C u0 p0 c0 {3,T} {11,S}
6  *6 C u0 p0 c0 {4,T} {12,S}
7     H u0 p0 c0 {1,S}
8     H u0 p0 c0 {1,S}
9     H u0 p0 c0 {2,S}
10    H u0 p0 c0 {2,S}
11    H u0 p0 c0 {5,S}
12    H u0 p0 c0 {6,S}
"""

ADJ_12_SHIFT_C = """
multiplicity 2
1  *2 C u0 p0 c0 {2,S} {3,S} {8,S} {9,S}
2  *1 C u0 p0 c0 {1,S} {10,S} {11,S} {12,S}
3  *3 C u1 p0 c0 {1,S} {4,S} {5,S}
4     C u0 p0 c0 {3,S} {6,D} {13,S}
5     C u0 p0 c0 {3,S} {7,D} {14,S}
6     C u0 p0 c0 {4,D} {7,S} {15,S}
7     C u0 p0 c0 {5,D} {6,S} {16,S}
8     H u0 p0 c0 {1,S}
9     H u0 p0 c0 {1,S}
10    H u0 p0 c0 {2,S}
11    H u0 p0 c0 {2,S}
12    H u0 p0 c0 {2,S}
13    H u0 p0 c0 {4,S}
14    H u0 p0 c0 {5,S}
15    H u0 p0 c0 {6,S}
16    H u0 p0 c0 {7,S}
"""

ADJ_EXO_SCISS = """
multiplicity 2
1  *3 C u0 p0 c0 {2,S} {8,S} {11,S} {12,S}
2  *2 C u0 p0 c0 {1,S} {3,B} {4,B}
3     C u0 p0 c0 {2,B} {5,B} {13,S}
4     C u0 p0 c0 {2,B} {7,B} {17,S}
5     C u0 p0 c0 {3,B} {6,B} {14,S}
6     C u0 p0 c0 {5,B} {7,B} {15,S}
7     C u0 p0 c0 {4,B} {6,B} {16,S}
8  *1 C u1 p0 c0 {1,S} {9,S} {18,S}
9     C u0 p0 c0 {8,S} {10,T}
10    C u0 p0 c0 {9,T} {19,S}
11    H u0 p0 c0 {1,S}
12    H u0 p0 c0 {1,S}
13    H u0 p0 c0 {3,S}
14    H u0 p0 c0 {5,S}
15    H u0 p0 c0 {6,S}
16    H u0 p0 c0 {7,S}
17    H u0 p0 c0 {4,S}
18    H u0 p0 c0 {8,S}
19    H u0 p0 c0 {10,S}
"""

ADJ_SUBST_S = """
multiplicity 2
1  *2 C u0 p0 c0 {3,S} {4,S} {5,S} {6,S}
2     C u0 p0 c0 {3,S} {7,S} {8,S} {9,S}
3  *3 C u1 p0 c0 {1,S} {2,S} {10,S}
4  *1 S u0 p2 c0 {1,S} {11,S}
5     H u0 p0 c0 {1,S}
6     H u0 p0 c0 {1,S}
7     H u0 p0 c0 {2,S}
8     H u0 p0 c0 {2,S}
9     H u0 p0 c0 {2,S}
10    H u0 p0 c0 {3,S}
11    H u0 p0 c0 {4,S}
"""

ADJ_CO = """
1 *1  C u0 p1 c-1 {2,T}
2 *3  O u0 p1 c+1 {1,T}
"""

ADJ_ALLYL = """
multiplicity 2
1      C u0 p0 c0 {2,D} {7,S} {8,S}
2      C u0 p0 c0 {1,D} {3,S} {9,S}
3      C u0 p0 c0 {2,S} {4,S} {10,S} {11,S}
4  *2  C u1 p0 c0 {3,S} {5,S} {6,S}
5      H u0 p0 c0 {4,S}
6      H u0 p0 c0 {4,S}
7      H u0 p0 c0 {1,S}
8      H u0 p0 c0 {1,S}
9      H u0 p0 c0 {2,S}
10     H u0 p0 c0 {3,S}
11     H u0 p0 c0 {3,S}
"""

ADJ_NO2_ON_X = """
1 X  u0  p0 c0  {2,S}
2 N  u0  p0 c+1  {1,S} {3,D} {4,S}
3 O  u0  p2 c0  {2,D}
4 O  u0  p3 c-1  {2,S}
"""
ADJ_X = "1 X  u0 p0 c0"

# Products of Surface_Dissociation_Charge_Separation (X-N=O + X-O)
ADJ_X_NO = """
1 X  u0 p0 c0 {2,S}
2 N  u0 p1 c0 {1,S} {3,D}
3 O  u0 p2 c0 {2,D}
"""
ADJ_X_O = """
1 X  u0 p0 c0 {2,D}
2 O  u0 p2 c0 {1,D}
"""


def mol(text):
    return Molecule().from_adjacency_list(text)


def _label_fingerprint(s):
    """
    Order-independent structural fingerprint of a product's labeled atoms.

    For each labeled atom, record (element, radical_electrons, charge) and the
    sorted list of bond orders to its neighbors. This lets the rmgpu side verify
    "label X sits on a Y atom with Z radical electrons / charge, bonded by
    orders [...]" without depending on RMG's internal atom ordering.
    """
    fp = {}
    for atom in s.atoms:
        if not atom.label:
            continue
        orders = sorted(round(bond.order, 6) for bond in atom.edges.values())
        fp[atom.label] = {
            'element': atom.element.symbol,
            'radical': int(atom.radical_electrons),
            'charge': int(atom.charge),
            'bonds': orders,
        }
    return fp


def summarize(structures):
    """Summarize a list of RMG Molecule/structures to JSON-safe data."""
    out = []
    for s in structures:
        out.append({
            'num_atoms': len(s.atoms),
            'label_fingerprint': _label_fingerprint(s),
            'net_charge': s.get_net_charge(),
        })
    return out


def family_meta(fam):
    """
    The family fields rmgpu's apply_recipe needs (RMG self.*), plus the
    *effective* product/reactant counts RMG's apply_recipe resolves to:
    RMG uses ``self.product_num or len(forward_template.products)`` for the
    forward direction and ``self.reactant_num or len(reverse_template.products)``
    for the reverse. Those effective counts are what the rmgpu port must
    pass as ``product_num`` (the engine is a free function with no template).
    """
    effective_forward = fam.product_num or len(fam.forward_template.products)
    reverse_template = fam.reverse_template
    effective_reverse = (fam.reactant_num
                         or (len(reverse_template.products)
                             if reverse_template is not None
                             else len(fam.forward_template.reactants)))
    return {
        'recipe': [list(a) for a in fam.forward_recipe.actions],
        'own_reverse': bool(fam.own_reverse),
        'reverse_map': fam.reverse_map,
        'product_num': fam.product_num,
        'reactant_num': fam.reactant_num,
        'electrons': int(fam.electrons),
        'effective_product_num_forward': int(effective_forward),
        'effective_product_num_reverse': int(effective_reverse),
    }


def main():
    database = KineticsDatabase()
    database.load_families(
        path=os.path.join(settings["test_data.directory"],
                          "testing_database/kinetics/families"),
        families=[
            "intra_H_migration",
            "R_Addition_MultipleBond",
            "H_Abstraction",
            "Intra_ene_reaction",
            "6_membered_central_C-C_shift",
            "1,2_shiftC",
            "Intra_R_Add_Exo_scission",
            "intra_substitutionS_isomerization",
            "R_Addition_COm",
            "Surface_Dissociation_Charge_Separation",
        ],
    )
    fams = database.families

    cases = {}

    def add(name, reactant_adjs, structures, forward=True, family=None):
        entry = {
            'reactants': reactant_adjs,
            'forward': forward,
            'products': summarize(structures),
        }
        if family is not None:
            entry.update(family_meta(family))
        cases[name] = entry

    # 1. H_Abstraction: acetaldehyde + H -> H2 + acetyl
    #    NOTE: the reverse direction is not exercised here: apply_recipe(forward=False)
    #    requires reactant atoms labeled for the REVERSE template (the products of a
    #    forward application carry the relabeled forward labels, so feeding them back
    #    raises ActionError in RMG-Py itself). The reverse direction is captured in
    #    the Surface_Dissociation_Charge_Separation case below.
    reactants = [mol(ADJ_H_ABSTRACTION_DONOR), mol(ADJ_H_RADICAL)]
    prods = fams["H_Abstraction"].apply_recipe(reactants)
    add("H_Abstraction", [ADJ_H_ABSTRACTION_DONOR, ADJ_H_RADICAL], prods,
        family=fams["H_Abstraction"])

    # 2. R_Addition_MultipleBond: benzene + H -> cyclohexadienyl radical
    reactants = [mol(ADJ_BENZENE), mol(ADJ_H_RADICAL)]
    prods = fams["R_Addition_MultipleBond"].apply_recipe(reactants)
    add("R_Addition_MultipleBond_benzene", [ADJ_BENZENE, ADJ_H_RADICAL], prods,
        family=fams["R_Addition_MultipleBond"])

    # 3. intra_H_migration (self-reverse; label swaps)
    prods = fams["intra_H_migration"].apply_recipe([mol(ADJ_INTRA_H_MIGRATION)])
    add("intra_H_migration", [ADJ_INTRA_H_MIGRATION], prods,
        family=fams["intra_H_migration"])

    # 4. Intra_ene_reaction
    prods = fams["Intra_ene_reaction"].apply_recipe([mol(ADJ_INTRA_ENE)])
    add("Intra_ene_reaction", [ADJ_INTRA_ENE], prods,
        family=fams["Intra_ene_reaction"])

    # 5. 6_membered_central_C-C_shift
    prods = fams["6_membered_central_C-C_shift"].apply_recipe([mol(ADJ_6MEM_SHIFT)])
    add("6_membered_central_C-C_shift", [ADJ_6MEM_SHIFT], prods,
        family=fams["6_membered_central_C-C_shift"])

    # 6. 1,2_shiftC
    prods = fams["1,2_shiftC"].apply_recipe([mol(ADJ_12_SHIFT_C)])
    add("1,2_shiftC", [ADJ_12_SHIFT_C], prods, family=fams["1,2_shiftC"])

    # 7. Intra_R_Add_Exo_scission
    prods = fams["Intra_R_Add_Exo_scission"].apply_recipe([mol(ADJ_EXO_SCISS)])
    add("Intra_R_Add_Exo_scission", [ADJ_EXO_SCISS], prods,
        family=fams["Intra_R_Add_Exo_scission"])

    # 8. intra_substitutionS_isomerization
    prods = fams["intra_substitutionS_isomerization"].apply_recipe([mol(ADJ_SUBST_S)])
    add("intra_substitutionS_isomerization", [ADJ_SUBST_S], prods,
        family=fams["intra_substitutionS_isomerization"])

    # 9. R_Addition_COm: CO + allyl
    reactants = [mol(ADJ_CO), mol(ADJ_ALLYL)]
    prods = fams["R_Addition_COm"].apply_recipe(reactants)
    add("R_Addition_COm", [ADJ_CO, ADJ_ALLYL], prods,
        family=fams["R_Addition_COm"])

    # 10. Surface_Dissociation_Charge_Separation (surface family; X sites).
    #     Reference only - rmgpu does not support surface sites (job-12).
    fam = fams["Surface_Dissociation_Charge_Separation"]
    exp_reactants = [mol(ADJ_NO2_ON_X), mol(ADJ_X)]
    exp_products = [mol(ADJ_X_NO), mol(ADJ_X_O)]
    labeled_rxn = Reaction(reactants=exp_reactants, products=exp_products)
    fam.add_atom_labels_for_reaction(labeled_rxn)
    labeled_reactant_mols = [m.molecule[0] for m in labeled_rxn.reactants]
    fam_products = fam.apply_recipe(labeled_reactant_mols)
    # Record the *labeled* reactants (as produced by add_atom_labels_for_reaction)
    # so the stored reactant adjacency lists are the ones actually applied.
    add("Surface_Dissociation_Charge_Separation_fwd",
        [m.to_adjacency_list() for m in labeled_reactant_mols], fam_products,
        family=fam)
    fam_reactants = fam.apply_recipe(fam_products, forward=False)
    add("Surface_Dissociation_Charge_Separation_rev",
        [m.to_adjacency_list() for m in fam_products], fam_reactants,
        forward=False, family=fam)

    # 11. Recipe reverse-recipe object (H_Abstraction)
    cases["H_Abstraction_recipe_fwd"] = [list(a) for a in
                                         fams["H_Abstraction"].forward_recipe.actions]
    cases["H_Abstraction_recipe_rev"] = [list(a) for a in
                                         fams["H_Abstraction"].forward_recipe.get_reverse().actions]

    # 12. intra_H_migration forward recipe (for the relabel test)
    cases["intra_H_migration_recipe"] = [list(a) for a in
                                         fams["intra_H_migration"].forward_recipe.actions]

    # 13. Recipe data from the real RMG-database families (as stored in rmgdb),
    #     so the rmgpu tests cover families with non-trivial reverse maps.
    cases["Br_Abstraction_recipe"] = [
        ['BREAK_BOND', '*1', 1, '*2'], ['FORM_BOND', '*2', 1, '*3'],
        ['GAIN_RADICAL', '*1', '1'], ['LOSE_RADICAL', '*3', '1']]
    cases["Br_Abstraction_reverse_map"] = {'*1': '*3', '*3': '*1'}

    out_path = os.path.join(REPO, "gates", "baselines", "job05",
                            "step01_apply_recipe_reference.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cases, f, indent=2)
    print("wrote", out_path)
    for name, value in cases.items():
        if isinstance(value, dict) and 'products' in value:
            print(f"  {name}: {len(value['products'])} product(s); "
                  f"net_charges={[p['net_charge'] for p in value['products']]}")
        else:
            print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
