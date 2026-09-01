import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule import group as G
adj = '1 *2 R!H u0 {2,[S,D,T,B]} {3,S}\n2 *1 R!H u1 {1,[S,D,T,B]}\n3 *3 H   u0 {1,S}\n'
grp = G.parse_group_adjlist_full(adj)
print("group atoms:")
for i,a in enumerate(grp.atoms):
    print("  n%d label=%s type=%s rad=%s chg=%s lp=%s" % (i, a.label, [t.label for t in a.atomtype], a.radical_electrons, a.charge, a.lone_pairs))
print("group edges:", {k: v.orders for k,v in grp.edges.items()})

mol = Molecule(smiles='C[CH]CCC')
feats, adjmat, types = G._molecule_explicit_graph(mol)
print("\nmol C[CH]CCC explicit (%d atoms):" % len(feats))
for i,(f,ty) in enumerate(zip(feats,types)):
    print("  atom%d %s type=%s rad=%d chg=%d lp=%s nbrs=%s" % (
        i, f['symbol'], ty, f['radical'], f['charge'], f['lone_pairs'],
        [(j,round(o,2)) for j,o in adjmat[i]]))
# check R!H specific-case for the radical C and the u0 C
r1h = G.atom_type('R!H')
print("\nR!H is atomtype:", r1h is not None)
for ty in set(types):
    t = G.atom_type(ty)
    if t is not None:
        print("  %s in R!H.specific: %s" % (ty, ty in r1h.specific))
    else:
        print("  %s NOT in atomtype tree" % ty)
