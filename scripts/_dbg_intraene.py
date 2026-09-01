import sys, json
sys.path.insert(0,'/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import template as T
REF='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step03_templates_reference.json'
S2='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step02_products_reference.json'
ref=json.load(open(REF)); s2=json.load(open(S2))
s2_by_fam={c['family']:c for c in s2['cases']}
fams={fl:T.TemplateFamily.from_references(ref['families'][fl],s2_by_fam[fl]) for fl in ref['families']}
ci,ri=26,0
case=s2['cases'][ci]; r=case['reactions'][ri]
print("family",case['family'],"reactants:",case['reactant_smiles'])
# build from SMILES (as the test does)
rm=[Molecule(smiles=s) for s in case['reactant_smiles']]
pm=[Molecule.from_adjacency_list(a) for a in r['products']]
print("product count:",len(pm))
fam=fams[case['family']]
print("reactant_num",fam.reactant_num,"ntop",len(fam.top),"dedup",[e.label for e in T._dedup_top(fam)])
# explicit-H reactants
print("base n atoms:",[m._rdkit.GetNumAtoms() for m in rm])
print("explicit n atoms:",[T._as_explicit_h(m)._rdkit.GetNumAtoms() for m in rm])
# run match with traceback
import traceback
try:
    # replicate unimolecular branch
    reactants=rm; products=pm
    graph=T._explicit_graph(reactants[0])
    print("graph atoms:",len(graph['atoms']))
    for g in T._slot_groups(fam,0):
        ms=T.match_explicit_graph(graph,g)
        labs=T.TemplateMatcher._labelings(g,ms)
        print("group",len(g.atoms),"atoms, mappings",len(ms))
        for labm in labs[:3]:
            ok=T._apply_and_check(fam,reactants,[labm],products)
            print("  lab",labm,"-> apply&check",ok)
            if not ok:
                # show recipe failure detail
                from rmgpu.core.recipe import apply_recipe, ActionError, KekulizationError
                c=T._as_explicit_h(reactants[0])
                for idx,lab in labm.items():
                    c._rdkit.GetAtomWithIdx(int(idx)).SetProp('label',str(lab))
                try:
                    prod=apply_recipe([c],fam.recipe,family_label=fam.label,forward=True,relabel_atoms=True,
                        own_reverse=fam.own_reverse,reverse_map=fam.reverse_map,
                        product_num=fam.product_num_forward,electrons=fam.electrons)
                    print("    recipe produced:",prod)
                    print("    expected products:",[p.get_formula() for p in pm])
                except Exception as ex:
                    print("    recipe EXC:",type(ex).__name__,str(ex)[:300])
        break
except Exception:
    traceback.print_exc()
