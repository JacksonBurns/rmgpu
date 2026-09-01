import sys, json
sys.path.insert(0,'/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule import group as G
from rmgpu.core import template as T
from rmgpu.core import recipe as R
from rmgpu.core import enumeration as enum
REF='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step03_templates_reference.json'
S2='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step02_products_reference.json'
ref=json.load(open(REF)); s2=json.load(open(S2))
s2_by_fam={c['family']:c for c in s2['cases']}
fams={fl:T.TemplateFamily.from_references(ref['families'][fl],s2_by_fam[fl]) for fl in ref['families']}
ci,ri=26,0
case=s2['cases'][ci]; r=case['reactions'][ri]; fam=fams[case['family']]
smi=case['reactant_smiles'][0]
pm=[Molecule.from_adjacency_list(a) for a in r['products']]
print("expected products:", [p.get_formula() for p in pm])

# Normalize a SMILES-built reactant to explicit-H via adjlist roundtrip, match + recipe
m=Molecule(smiles=smi)
norm=Molecule.from_adjacency_list(m.to_adjlist())
print("norm n atoms:", norm._rdkit.GetNumAtoms(), "radical idx:", [i for i,a in enumerate(norm._rdkit.GetAtoms()) if a.GetNumRadicalElectrons()])
gr=G._explicit_graph(norm)
g=fam.entries['1_3_unsaturated_pentane_backbone'].item
ms=T.match_explicit_graph(gr,g)
labs=T.TemplateMatcher._labelings(g,ms)
print("n mappings:",len(ms))
ok=False
for labm in labs:
    c=Molecule.from_adjacency_list(norm.to_adjlist())
    for idx,lb in labm.items():
        c._rdkit.GetAtomWithIdx(int(idx)).SetProp('label',str(lb))
    try:
        prod=R.apply_recipe([c],fam.recipe,family_label=fam.label,forward=True,relabel_atoms=True,
            own_reverse=fam.own_reverse,reverse_map=fam.reverse_map,
            product_num=fam.product_num_forward,electrons=fam.electrons)
        if prod and enum.same_species_lists(pm,prod,strict=False):
            ok=True
            print("  MATCH via normalization, product:",[p.get_formula() for p in prod])
    except Exception as e:
        print("  EXC",type(e).__name__,str(e)[:120])
print("SMILES-built + normalization ->", ok)
