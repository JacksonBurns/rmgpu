import sys, json
sys.path.insert(0,'/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule import group as G
from rmgpu.molecule.adjlist import parse_adjlist
S2='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step02_products_reference.json'
s2=json.load(open(S2))
case=s2['cases'][26]  # Intra_ene C[CH]C1=CC=CC=C1
print("family",case['family'],case['reactant_smiles'])
app=case['applications'][0]
lr=app['labeled_reactants'][0]
print("recorded labeled reactant (first 300):")
print(lr[:300])
# parse recorded atom order (symbols + labels)
parsed = parse_adjlist(lr)
atoms = parsed[-1]  # last element is the atoms list
rec_order = [(a['symbol'], a['label']) for a in atoms]
print("recorded atom order (symbol,label):")
print(rec_order)
# my _explicit_graph order for the SAME molecule built from SMILES
m=Molecule(smiles=case['reactant_smiles'][0])
g=G._explicit_graph(m)
my_order=[(a['symbol'], a.get('atomtype','')) for a in g['atoms']]
print("my _explicit_graph order (symbol,atomtype):")
print(my_order)
# the label positions in recorded: which atom indices have labels?
rec_labels={i:lbl for i,(s,lbl) in enumerate(rec_order) if lbl}
print("recorded label indices:", rec_labels)
# now build the molecule from the recorded adjlist (RMG ordering) and see its _explicit_graph order
m2=Molecule.from_adjacency_list(lr)
g2=G._explicit_graph(m2)
print("from-adjlist _explicit_graph order (symbol):")
print([a['symbol'] for a in g2['atoms']])
print("n atoms from-adjlist explicit:", len(g2['atoms']))
