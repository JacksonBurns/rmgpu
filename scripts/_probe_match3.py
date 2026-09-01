"""Debug a single overcounting group match."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule import group as G

# component 7 of H_Abstraction slot0 (X_H leaf): "1 *1 R u0 {2,S} 2 *2 H u0 {1,S}"
adj = "1 *1 R u0 {2,S}\n2 *2 H u0 {1,S}\n"
grp = G.parse_group_adjlist_full(adj)
print("group atoms:", [(a.label, [t.label for t in a.atomtype], a.radical_electrons, a.charge, a.lone_pairs) for a in grp.atoms])
print("group edges:", {k: v.orders for k, v in grp.edges.items()})

mol = Molecule(smiles='C')  # methane
feats, adjmat, types = G._molecule_explicit_graph(mol)
print("\nmethane explicit:", len(feats), "atoms")
for i,(f,ty) in enumerate(zip(feats, types)):
    print("  atom%d %s type=%s rad=%d chg=%d lp=%s" % (i, f['symbol'], ty, f['radical'], f['charge'], f['lone_pairs']))
print("R atomtype specific count:", len(G.atom_type('R').specific))

maps = G.match_group(mol, grp)
print("\nn mappings:", len(maps))
print("first 8:", maps[:8])
# Show the mol atom indices used
used = set()
for m in maps:
    used.add(m[1])
print("mol atoms used for group H (index1):", sorted(used))
