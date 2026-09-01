import sys, json
sys.path.insert(0,'/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule import group as G
from rmgpu.core import template as T
from rmgpu.core import recipe as R
REF='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step03_templates_reference.json'
S2='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step02_products_reference.json'
ref=json.load(open(REF)); s2=json.load(open(S2))
fam=T.TemplateFamily.from_references(ref['families']['Intra_ene_reaction'],
    next(c for c in s2['cases'] if c['family']=='Intra_ene_reaction'))
smi='C[CH]C1=CC=CC=C1'
m=Molecule(smiles=smi)
graph=G._explicit_graph(m)
g=fam.entries['1_3_unsaturated_pentane_backbone'].item
ms=T.match_explicit_graph(graph,g)
lab=T.TemplateMatcher._labelings(g,ms)[0]
print("labeling (mol_idx->label):",lab)
# build canonical adjlist form, set labels
c=Molecule.from_adjacency_list(m.to_adjlist())
for idx,lb in lab.items():
    c._rdkit.GetAtomWithIdx(int(idx)).SetProp('label',str(lb))
# print the labeled atom identities
for a in c._rdkit.GetAtoms():
    if a.HasProp('label'):
        print("  label",a.GetProp('label'),"atom",a.GetIdx(),a.GetSymbol(),"rad",a.GetNumRadicalElectrons())
print("\nrecipe actions:")
for act in fam.recipe.actions:
    print("  ",act)
# apply action by action to trace where the ring breaks
rw=c._rdkit
# We'll just call apply_forward once and print result, then manually apply each bond change
print("\nFull apply_recipe result:")
try:
    prod=R.apply_recipe([c],fam.recipe,family_label=fam.label,forward=True,relabel_atoms=True,
        own_reverse=fam.own_reverse,reverse_map=fam.reverse_map,
        product_num=fam.product_num_forward,electrons=fam.electrons)
    for p in (prod or []):
        print("  product formula:", p.get_formula(), "smiles:", p.to_smiles())
except Exception as e:
    print("  EXC",type(e).__name__,str(e)[:200])
