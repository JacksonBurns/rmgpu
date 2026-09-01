import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.core.recipe import ReactionRecipe, apply_recipe
from rmgpu.molecule.molecule import Molecule

# H_Abstraction: C + [H] -> H2 + CH3
donor = """
1 *1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2 *2 H u0 p0 c0 {1,S}
3    H u0 p0 c0 {1,S}
4    H u0 p0 c0 {1,S}
5    H u0 p0 c0 {1,S}
"""
radical = "1 *3 H u1 p0 c0"
recipe = ReactionRecipe.from_data([
    ['BREAK_BOND', '*1', 1, '*2'],
    ['FORM_BOND', '*2', 1, '*3'],
    ['GAIN_RADICAL', '*1', '1'],
    ['LOSE_RADICAL', '*3', '1'],
])
m1 = Molecule.from_adjacency_list(donor)
m2 = Molecule.from_adjacency_list(radical)
# set atomid props
ids1 = [10, 11, 12, 13, 14]
ids2 = [20]
for i, a in enumerate(m1._rdkit.GetAtoms()):
    a.SetProp('atomid', str(ids1[i]))
for i, a in enumerate(m2._rdkit.GetAtoms()):
    a.SetProp('atomid', str(ids2[i]))
prods = apply_recipe([m1, m2], recipe, family_label='h_abstraction',
                     forward=True, own_reverse=True, reverse_map=None,
                     product_num=2, electrons=0)
print("n products:", len(prods))
for p in prods:
    ids = [a.GetProp('atomid') if a.HasProp('atomid') else '?' for a in p._rdkit.GetAtoms()]
    labels = [a.GetProp('label') if a.HasProp('label') else '' for a in p._rdkit.GetAtoms()]
    print("  piece:", p.get_formula(), "ids:", ids, "labels:", labels)
