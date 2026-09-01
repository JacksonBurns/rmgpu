import sys
sys.path.insert(0,'/home/jackson/rmgpu/rmgpu')
import json
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import template as T

REF='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step03_templates_reference.json'
S2='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step02_products_reference.json'
ref=json.load(open(REF)); s2=json.load(open(S2))
s2_by_family={}
for c in s2['cases']: s2_by_family.setdefault(c['family'],c)
fam=T.TemplateFamily.from_references(ref['families']['H_Abstraction'], s2_by_family['H_Abstraction'])

# H_Abstraction ci=0 ri=0: C + [H] -> CH4
c=s2['cases'][0]; r=c['reactions'][0]
rmols=[Molecule.from_adjacency_list(a) for a in r['reactants']]
pmols=[Molecule.from_adjacency_list(a) for a in r['products']]
print("reactants:", [m.get_formula() for m in rmols])
print("products:", [m.get_formula() for m in pmols])

matcher=T.TemplateMatcher(fam)
# slot0 = X_H_or... (the R-H donor), slot1 = Y_rad... (the radical)
ma=matcher.match_molecule(rmols[0],0,'ab')
mb=matcher.match_molecule(rmols[1],1,'ab')
print("slot0 labelings on reactant0 (C):", ma)
print("slot1 labelings on reactant1 ([H]):", mb)
# The template is [X_H(=the C-H), Y_rad(=the radical)]. For C+[H] -> CH4,
# X_H should match C (the H-donor: C-H bond), Y_rad matches [H].
# But which reactant is which? branch ab: reactant0->slot0, reactant1->slot1.
# Let's also check ba
ma2=matcher.match_molecule(rmols[0],0,'ba')
mb2=matcher.match_molecule(rmols[1],1,'ba')
print("branch ba: slot0 on reactant0:", ma2, " slot1 on reactant1:", mb2)

# Now test _apply_and_check with the right labeling
for lab_a in ma:
    for lab_b in mb:
        print("trying lab", lab_a, lab_b)
        ok=T._apply_and_check(fam, rmols, [lab_a, lab_b], pmols)
        print("  ->", ok)
