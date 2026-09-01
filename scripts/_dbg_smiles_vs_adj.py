import sys, json
sys.path.insert(0,'/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule import group as G
from rmgpu.core import template as T
REF='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step03_templates_reference.json'
S2='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step02_products_reference.json'
ref=json.load(open(REF)); s2=json.load(open(S2))
s2_by_fam={c['family']:c for c in s2['cases']}
fams={fl:T.TemplateFamily.from_references(ref['families'][fl],s2_by_fam[fl]) for fl in ref['families']}
ci,ri=26,0
case=s2['cases'][ci]; r=case['reactions'][ri]
fam=fams[case['family']]
smi=case['reactant_smiles'][0]

def diag(tag, rm, pm):
    print("=== %s ==="%tag)
    m=rm[0]
    print("  n _rdkit atoms:", m._rdkit.GetNumAtoms(), "(has H:", any(a.GetSymbol()=='H' for a in m._rdkit.GetAtoms()),")")
    # explicit graph ordering
    gr=G._explicit_graph(m)
    print("  explicit n:", len(gr['atoms']), "symbols:", [a['symbol'] for a in gr['atoms']])
    # to_adjlist roundtrip ordering
    adj=m.to_adjlist()
    m2=Molecule.from_adjacency_list(adj)
    gr2=G._explicit_graph(m2)
    print("  roundtrip explicit n:", len(gr2['atoms']), "symbols:", [a['symbol'] for a in gr2['atoms']])
    rxn=T.Reaction(list(rm),list(pm))
    got=T.match(fam,rxn)
    print("  MATCH:", got)
    return got

# SMILES-built
rm_s=[Molecule(smiles=smi)]
pm_s=[Molecule.from_adjacency_list(a) for a in r['products']]
g1=diag("SMILES-built", rm_s, pm_s)

# adjlist-built (harness path)
rm_a=[Molecule.from_adjacency_list(a) for a in r['reactants']]
pm_a=[Molecule.from_adjacency_list(a) for a in r['products']]
g2=diag("adjlist-built", rm_a, pm_a)
print("\ngot equal:", g1==g2)
