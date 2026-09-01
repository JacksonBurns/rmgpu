import sys, json
sys.path.insert(0,'/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import template as T
REF='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step03_templates_reference.json'
S2='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step02_products_reference.json'
ref=json.load(open(REF)); s2=json.load(open(S2))
s2_by_family={}
for c in s2['cases']: s2_by_family.setdefault(c['family'],c)
fams={}
for fl in ref['families']:
    fams[fl]=T.TemplateFamily.from_references(ref['families'][fl], s2_by_family[fl])

targets=[('R_Recombination',14,0),('Intra_ene_reaction',26,0),('1,2_shiftC',27,0),('Singlet_Val6_to_triplet',29,0)]
rxn_map={}
for ci,c in enumerate(s2['cases']):
    for ri,r in enumerate(c['reactions']):
        rxn_map[(ci,ri)]=([Molecule.from_adjacency_list(a) for a in r['reactants']],
                          [Molecule.from_adjacency_list(a) for a in r['products']])
for (fl,ci,ri) in targets:
    rmols,pmols=rxn_map[(ci,ri)]
    fam=fams[fl]
    print("="*50)
    print(fl, "ci=%d ri=%d"%(ci,ri))
    print("  reactants:", [m.get_formula() for m in rmols], " reactant_num=%s"%fam.reactant_num, " ntop=%d"%len(fam.top))
    # reactant count guard?
    if len(rmols)!=fam.reactant_num:
        print("  !! reactant count mismatch: %d vs %d"%(len(rmols),fam.reactant_num))
    # matcher labelings
    matcher=T.TemplateMatcher(fam)
    dedup=T._dedup_top(fam)
    print("  dedup_top:", [e.label for e in dedup])
    if len(rmols)==1:
        graph=T._explicit_graph(rmols[0])
        for si,e in enumerate(dedup):
            gs=T._slot_groups(fam, si) if si < len(fam.top) else []
        # unimolecular: slot 0
        for g in T._slot_groups(fam,0):
            ms=T.match_explicit_graph(graph,g)
            labs=T.TemplateMatcher._labelings(g,ms)
            print("  slot0 group %d atoms %d labelings"%[len(g.atoms),len(ms)])
            if labs:
                print("    first labeling:", labs[0])
                ok=T._apply_and_check(fam, rmols, [labs[0]], pmols)
                print("    apply_and_check:", ok)
                if not ok:
                    # show what recipe does
                    import traceback
                    c0=rmols[0].copy(clear_labels=True)
                    T._set_labels_on_mol(c0, labs[0])
                    try:
                        prod=T.apply_recipe([c0], fam.recipe, family_label=fam.label, forward=True,
                            relabel_atoms=True, own_reverse=fam.own_reverse, reverse_map=fam.reverse_map,
                            product_num=fam.product_num_forward, electrons=fam.electrons)
                        print("    recipe products:", [p.get_formula() for p in prod] if prod else prod)
                        print("    expected:", [p.get_formula() for p in pmols])
                    except Exception as ex:
                        print("    recipe EXC:", type(ex).__name__, str(ex)[:200])
