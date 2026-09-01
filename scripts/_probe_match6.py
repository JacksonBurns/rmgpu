import json, sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule import group as G
from rmgpu.molecule.atomtype import assign_atom_types
p3 = '/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step03_templates_reference.json'
d3 = json.load(open(p3))
for c in d3['match_cases']:
    if c['family']=='intra_H_migration':
        smiles = c['reactant_smiles'][0]
        sg = c['slot_groups_adj'][0]
        m = Molecule(smiles=smiles)
        types = assign_atom_types(m)
        print("molecule", smiles, "atom types:", types)
        for i,a in enumerate(sg):
            g = G.parse_group_adjlist(a)
            ns = G.match_group(m, g)
            print(f"comp {i}: n={len(ns)}  first={ns[:1]}")
        break
