import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import enumeration as enum
from rmgpu.core.recipe import (_merge_molecules, _re_aromatize, _kekulize_piece,
                               _dof_kekulize, _all_six_rings, KekulizationError,
                               ReactionRecipe)
import json

ref = json.load(open('gates/baselines/job05/step02_products_reference.json'))
case = None
for c in ref['cases']:
    if c['family']=='R_Addition_MultipleBond' and c['reactant_smiles']==['C1=CC=CC=C1','[H]']:
        case = c; break

fam = enum.Family.from_reference(case)
reactants = [Molecule(smiles=s) for s in case['reactant_smiles']]
fam.matcher = enum.RecordedMatcher(case)

# Build the labeled reactants for the first application, merge, re-aromatize,
# apply the recipe, and inspect the ring bond orders.
def build_labeled(app):
    mols = []
    for adj in app['reactants']:
        m = Molecule.from_adjacency_list(adj)
        mols.append(m)
    return mols

app = case['applications'][0]
labeled = build_labeled(app)
merged = _merge_molecules(labeled)
_re_aromatize(merged)
recipe = ReactionRecipe(app['recipe'])
recipe.apply_forward(merged, True)
print("after recipe, ring bond orders:")
for b in merged.GetBonds():
    o = b.GetBondTypeAsDouble()
    if o in (1.5, 0.5, 2.5) or not (o == int(o)):
        a1 = merged.GetAtomWithIdx(b.GetBeginAtomIdx())
        a2 = merged.GetAtomWithIdx(b.GetEndAtomIdx())
        print("  %s-%s order=%.1f aro=%s/%s" % (a1.GetSymbol(), a2.GetSymbol(), o, a1.GetIsAromatic(), a2.GetIsAromatic()))
rings = _all_six_rings(merged)
print("6-rings:", rings)
m2 = Chem.RWMol(merged)
ok = _dof_kekulize(m2)
print("dof_kekulize ->", ok)
if ok:
    print("SMILES after dof:", Chem.MolToSmiles(m2, canonical=True))
# also try kekulize_piece
m3 = Chem.RWMol(merged)
try:
    _kekulize_piece(m3)
    print("kekulize_piece SMILES:", Chem.MolToSmiles(m3, canonical=True))
    for a in m3.GetAtoms():
        print("  atom", a.GetIdx(), a.GetSymbol(), "arom=", a.GetIsAromatic())
except KekulizationError as e:
    print("kekulize_piece RAISED:", e)
