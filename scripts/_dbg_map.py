import sys, json
sys.path.insert(0,'/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule import group as G
from rmgpu.core import template as T
REF='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step03_templates_reference.json'
S2='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step02_products_reference.json'
ref=json.load(open(REF)); s2=json.load(open(S2))
fam=T.TemplateFamily.from_reference(ref['families']['Intra_ene_reaction'])
# the forward group (1_3_unsaturated_pentane_backbone)
g=fam.entries['1_3_unsaturated_pentane_backbone'].item
print("group atoms:")
for i,a in enumerate(g.atoms):
    print("  gi=%d label=%r atomtype=%s rad=%s"%(i, a.label, [x.label for x in a.atomtype], a.radical_electrons))
print("group bonds:", {k:tuple(v.orders) for k,v in g.edges.items()})

smi='C[CH]C1=CC=CC=C1'
m=Molecule(smiles=smi)
graph=G._explicit_graph(m)
# atomtypes + positions of C (radical) and H
print("\ngraph atoms:")
for i,a in enumerate(graph['atoms']):
    print("  i=%d %s atype=%s rad=%d charge=%d"%(i,a['symbol'],a['atomtype'],a['radical'],a['charge']))
# mappings
ms=T.match_explicit_graph(graph,g)
print("\n%d mappings; first mapping group->mol:"%len(ms), ms[0])
# what mol atom is each group label on?
lab=T.TemplateMatcher._labelings(g,ms)[0]
print("labeling mol_idx->label:", lab)
