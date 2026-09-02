"""Probe: current rmgpu resonance forms + matcher behavior for 1-phenylethyl."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import generate_resonance_structures
from rmgpu.core.enumeration import expand_resonance, assign_fresh_ids
from rmgpu.molecule.group import _explicit_graph, match_explicit_graph, parse_group_adjlist_full

# The 1-phenylethyl radical, kekulized (as the gate drives it)
smi_kek = 'C[CH]C1=CC=CC=C1'
print('=== input (kekulized) ===', smi_kek)
mol = Molecule(smiles=smi_kek)
print('aromatic bonds in input:', any(b.GetIsAromatic() for b in mol._rdkit.GetBonds()))

# Raw resonance
forms = generate_resonance_structures(mol)
print('\n=== generate_resonance_structures: %d forms ===' % len(forms))
for i, f in enumerate(forms):
    print('  [%d] %s' % (i, f.to_smiles()))

# Expand + fresh ids (what the gate drives)
print('\n=== expand_resonance + assign_fresh_ids ===')
species = expand_resonance([mol])
assign_fresh_ids(species)
for i, f in enumerate(species[0]):
    ar = [b.GetBondTypeAsDouble() for b in f._rdkit.GetBonds() if b.GetIsAromatic()]
    print('  form[%d] %s  aromatic_bond_orders=%s' % (i, f.to_smiles(), ar))

# Now test the Intra_ene template against each form.
# Intra_ene_reaction template reactant group (from RMG database).
template_adjlist = """\
1 R u[1] {2,S}
2 R u[0] c[0] {3,D} {1,S}
3 R u[0] c[0] {2,D} {4,S}
4 R u[0] c[0] {3,S}
"""
grp = parse_group_adjlist_full(template_adjlist)
print('\n=== Intra_ene template reactant group ===')
print('  atoms=%d bonds=%d' % (len(grp.atoms), len(grp.edges)))
for (a, b), bond in grp.edges.items():
    print('  bond %d-%d orders=%s' % (a, b, bond.orders))

print('\n=== match count per form (match_molecule slot 0) ===')
for i, f in enumerate(species[0]):
    graph = _explicit_graph(f)
    matches = match_explicit_graph(graph, grp)
    print('  form[%d] %s  -> %d matchings' % (i, f.to_smiles(), len(matches)))
