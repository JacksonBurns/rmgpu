"""Harness: run rmgpu match() on all step-02 reactions for ALL 7 families,
compare verdicts (match/no-match) + labels against the recorded RMG-Py verdict
matrix (step03_match_verdicts.json). Reactions (reactants+products) are the
step-02 recorded adjlists (built via from_adjacency_list)."""
import json, sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import template as T

REF='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step03_templates_reference.json'
VERD='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step03_match_verdicts.json'
S2='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step02_products_reference.json'

ref=json.load(open(REF)); verd=json.load(open(VERD)); s2=json.load(open(S2))

s2_by_family={}
for c in s2['cases']:
    s2_by_family.setdefault(c['family'], c)
fams={}
for fl in verd['families']:
    try:
        fams[fl]=T.TemplateFamily.from_references(ref['families'][fl], s2_by_family[fl])
    except Exception as e:
        print("BUILD FAIL", fl, type(e).__name__, e)
        import traceback; traceback.print_exc()

# (ci,ri) -> (reactant_mols, product_mols)
rxn_map={}
for ci,c in enumerate(s2['cases']):
    for ri,r in enumerate(c['reactions']):
        try:
            rmols=[Molecule.from_adjacency_list(a) for a in r['reactants']]
            pmols=[Molecule.from_adjacency_list(a) for a in r['products']]
        except Exception as e:
            print("BUILD REACTION FAIL ci=%d ri=%d %s: %s"%(ci,ri,type(e).__name__,e))
            continue
        rxn_map[(ci,ri)]=(rmols,pmols)

agree_verd=dis_verd=0; disv=[]
agree_lab=dis_lab=0; disl=[]
n_checked=0
for rec in verd['reactions']:
    ci,ri,own=rec['ci'],rec['ri'],rec['family']
    if (ci,ri) not in rxn_map:
        continue
    rmols,pmols=rxn_map[(ci,ri)]
    for fl in verd['families']:
        if fl not in fams: continue
        n_checked+=1
        rxn=T.Reaction(list(rmols), list(pmols))
        try:
            lab=T.match(fams[fl], rxn)
        except Exception as e:
            lab=('EXC', type(e).__name__, e)
            import traceback; traceback.print_exc()
        my_matched = (lab is not None) and not isinstance(lab,tuple)
        my_labels = None if (lab is None or isinstance(lab,tuple)) else lab
        ref_v=verd['verdicts']['%s|%d|%d'%(fl,ci,ri)]
        ref_matched=ref_v['matched']
        ref_labels=ref_v['template']
        if my_matched==ref_matched:
            agree_verd+=1
        else:
            dis_verd+=1
            if len(disv)<30:
                disv.append((fl,ci,ri,'own' if fl==own else 'x','my=%s ref=%s'%(my_matched,ref_matched), str(my_labels)[:60]))
        if ref_matched and my_matched:
            if my_labels==ref_labels:
                agree_lab+=1
            else:
                dis_lab+=1
                if len(disl)<30:
                    disl.append((fl,ci,ri,'own' if fl==own else 'x','my=%s ref=%s'%(my_labels,ref_labels)))
print("checked", n_checked)
print("VERDICT agree", agree_verd, "disagree", dis_verd)
print("LABEL agree(both-match)", agree_lab, "disagree", dis_lab)
print("\n-- verdict disagreements --")
for x in disv: print("  ", x)
print("\n-- label disagreements --")
for x in disl: print("  ", x)
