import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.core.recipe import (ReactionRecipe, _merge_molecules, _re_aromatize,
                               _kekulize_piece, _is_benzene, _connected_pieces, apply_recipe)
from rmgpu.core import enumeration as enum
import json
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'gates', 'baselines', 'job05', 'step02_products_reference.json')
data = json.load(open(BASE))
case = [c for c in data['cases'] if c['family']=='R_Addition_MultipleBond' and c['reactant_smiles']==['C1=CC=CC=C1','[H]']][0]
print("RECIPE:", case['family_meta']['recipe'])
print("num_template_reactants_effective:", case['family_meta'].get('num_template_reactants_effective'))
match = enum.RecordedMatcher(case)
apps = list(match.yield_applications())
print("n apps:", len(apps))
structures = apps[0]
print("n structures:", len(structures))
for i,s in enumerate(structures):
    m = Chem.Mol(s._rdkit)
    print(f"  struct{i} atoms:", [(a.GetSymbol(), a.GetIsAromatic()) for a in m.GetAtoms()])
    print(f"  struct{i} bonds:", [(b.GetBeginAtomIdx(),b.GetEndAtomIdx(),b.GetBondTypeAsDouble()) for b in m.GetBonds()])
merged = _merge_molecules(structures)
print("\nMERGED (before re_aromatize) bond orders:", sorted(b.GetBondTypeAsDouble() for b in merged.GetBonds()))
_re_aromatize(merged)
print("MERGED (after re_aromatize) bond orders:", sorted(b.GetBondTypeAsDouble() for b in merged.GetBonds()))
recipe = ReactionRecipe.from_data(case['family_meta']['recipe'])
try:
    recipe._apply(merged, True, True)
    print("\nAFTER RECIPE bond orders:", sorted(b.GetBondTypeAsDouble() for b in merged.GetBonds()))
    print("AFTER RECIPE aromatic atoms:", [(a.GetIdx(),a.GetSymbol(),a.GetIsAromatic(),a.GetNumRadicalElectrons()) for a in merged.GetAtoms()])
    for b in merged.GetBonds():
        print(f"   bond {b.GetBeginAtomIdx()}={b.GetBeginAtomIdx()}-sym {merged.GetAtomWithIdx(b.GetBeginAtomIdx()).GetSymbol()} - {merged.GetAtomWithIdx(b.GetEndAtomIdx()).GetSymbol()} order={b.GetBondTypeAsDouble()} aromatic={b.GetIsAromatic()}")
except Exception as e:
    import traceback; traceback.print_exc()
