import json, sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule import group as G
from rmgpu.molecule.atomtype import assign_atom_types

# component 0 of intra_H_migration
adj = """1 *2 R!H u0 {2,[S,D,T,B]} {3,S}
2 *1 R!H u1 {1,[S,D,T,B]}
3 *3 H   u0 {1,S}
"""
g = G.parse_group_adjlist(adj)
print("group atoms:")
for i, a in enumerate(g.atoms):
    print(f"  {i}: label={a.label!r} atomtype={[t.label if hasattr(t,'label') else t for t in (a.atomtype or [])]} rad={a.radical_electrons} charge={a.charge} lp={a.lone_pairs}")
print("group edges:", {f"({k[0]},{k[1]})": v.orders for k, v in g.edges.items()})

m = Molecule(smiles='C[CH]CCC')
types = assign_atom_types(m)
feats, adjl, atypes = G._molecule_explicit_graph(m)
print("\nmolecule feats:")
for i in range(len(feats)):
    print(f"  {i}: {feats[i]['symbol']} rad={feats[i]['radical']} charge={feats[i]['charge']} type={atypes[i]}")

# feasibility of each group atom against each molecule atom
for gi in range(len(g.atoms)):
    row = []
    for mi in range(len(feats)):
        row.append(G._atom_matches_group_atom(feats[mi], g.atoms[gi], atypes[mi]))
    print(f"group atom {gi} feasible on mol atoms: {[i for i in range(len(feats)) if row[i]]}")
