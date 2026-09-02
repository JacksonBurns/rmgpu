"""Probe 4: exact aromatic state + R_Addition baseline matching."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.group import _explicit_graph, parse_group_adjlist_full
from rmgpu.core.template import TemplateMatcher, _slot_groups
from rmgpu.core.family import KineticsFamilies, Family
from rmgpu.core import enumeration as enum
from rdkit import Chem

def full_state(mol, label):
    m = mol._rdkit
    bond_ar = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                b.GetBondTypeAsDouble(), b.GetIsAromatic())
               for b in m.GetBonds() if b.GetIsAromatic()]
    print('  %s: smiles=%s n_aromatic_bonds=%d atom_aromatic=%d' % (
        label, mol.to_smiles(), len(bond_ar),
        sum(1 for a in m.GetAtoms() if a.GetIsAromatic())))
    for (i, j, o, ar) in bond_ar:
        print('     bond %d-%d order=%.1f isAromatic=%s' % (i, j, o, ar))

print('=== stored state: kekulized benzene input C1=CC=CC=C1 ===')
b = Molecule(smiles='C1=CC=CC=C1')
full_state(b, 'benzene_kek')

print('\n=== stored state: aromatic benzene input c1ccccc1 ===')
b2 = Molecule(smiles='c1ccccc1')
full_state(b2, 'benzene_ar')

# Now: does RDKit re-detect aromaticity on a fresh explicit-H copy?
print('\n=== re-aromatization test (fresh copy) ===')
for label, m in [('benzene_kek', b), ('benzene_ar', b2),
                 ('ortho_form', Molecule(smiles='CC=C1[CH]C=CC=C1'))]:
    em = Chem.AddHs(m._rdkit)
    try:
        Chem.SanitizeMol(em)
        Chem.SetAromaticity(em)
    except Exception as e:
        print('  %s: re-aromatize ERROR %s' % (label, e))
        continue
    ar_bonds = [b2.GetBondTypeAsDouble() for b2 in em.GetBonds() if b2.GetIsAromatic()]
    print('  %s: after SetAromaticity aromatic_bonds=%d' % (label, len(ar_bonds)))

# Current R_Addition_MultipleBond benzene matching baseline
print('\n=== R_Addition_MultipleBond benzene+H baseline ===')
kf = KineticsFamilies().load('default')
fam = {f.label: f for f in kf.families}['R_Addition_MultipleBond']
efam = enum.Family(
    label=fam.label, recipe=fam.recipe, reverse_recipe=fam.reverse_recipe,
    own_reverse=fam.own_reverse, reversible=fam.reversible,
    allow_charged_species=fam.allow_charged_species, electrons=fam.electrons,
    reactant_num_effective=fam.num_template_reactants_effective,
    product_num_forward=fam.product_num_forward,
    reverse_map=fam.reverse_map,
    template_labels=[e.label for e in fam.forward_template],
    forbidden=fam.forbidden,
)
matcher = TemplateMatcher(fam)
efam.matcher = matcher
benzene = Molecule(smiles='C1=CC=CC=C1')
h = Molecule(smiles='[H]')
# match benzene against slot 0 (R_R)
graph = _explicit_graph(benzene)
mappings = matcher.match_graph(graph, 0)
print('  benzene vs R_R (slot0): %d matchings' % len(mappings))
rxns = enum.generate_reactions(efam, [benzene, h], matcher=matcher)
for r in rxns:
    print('  rxn deg=%.1f products=%s' % (r.degeneracy,
           [p.to_smiles() for p in r.products]))
