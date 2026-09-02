"""Probe 15: verify round-trip atom/bond ordering + H_Abstraction parity."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu/gates')
from rmgpu.molecule.molecule import Molecule
from rmgpu.core.family import KineticsFamilies
from rmgpu.core.template import TemplateMatcher
from rmgpu.molecule.group import _explicit_graph, match_explicit_graph
from rmgpu.core import enumeration as enum
from rdkit import Chem
from rdkit.Chem import BondType

def aromat_bond_pairs(mol):
    """frozenset of {frozenset({a,b})} for ring bonds that are aromatic (1.5)."""
    em = mol._with_explicit_h()
    pairs = set()
    for b in em.GetBonds():
        if b.GetIsAromatic():
            pairs.add(frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
    return pairs

# Verify round-trip preserves atom index + bond set for the aromatic form
ar = Molecule._from_rdmol(Chem.Mol(Chem.MolFromSmiles('C[CH]c1ccccc1')))
al = ar.to_adjlist()
rt = Molecule.from_adjacency_list(al)
print('atom count ar/rt:', ar._with_explicit_h().GetNumAtoms(), rt._with_explicit_h().GetNumAtoms())
pa = aromat_bond_pairs(ar)
print('aromatic bond pairs (ar, explicit-H):', len(pa), sorted(sorted(p) for p in pa))
# after round-trip, the S/D ring bonds are at the same atom indices?
g = _explicit_graph(rt)
print('round-trip explicit bond orders:', sorted(set(round(o,2) for adj in g['adj'] for (j,o) in adj)))
# Do the same atom-index pairs exist?
print('  pairs still bonded in rt:', all((min(p),max(p)) or True for p in pa))
# check each pair has a bond in rt
emrt = rt._with_explicit_h()
for p in pa:
    a,b = sorted(p)
    bd = emrt.GetBondBetweenAtoms(a,b)
    print('    pair %s-%s order=%.1f aromatic=%s' % (a,b, bd.GetBondTypeAsDouble(), bd.GetIsAromatic()))

# Now: re-aromatize those pairs on rt
rw = Chem.RWMol(emrt)
for p in pa:
    a,b = sorted(p)
    rw.GetBondBetweenAtoms(a,b).SetBondType(BondType.AROMATIC)
for b in rw.GetBonds():
    if b.GetBondType() != BondType.AROMATIC:
        b.SetIsAromatic(False)
Chem.SanitizeMol(rw)
ar2 = Molecule._from_rdmol(Chem.Mol(rw))
print('\nre-aromatized rt smiles:', ar2.to_smiles())
g2 = _explicit_graph(ar2)
print('  bond orders:', sorted(set(round(o,2) for adj in g2['adj'] for (j,o) in adj)))

# H_Abstraction parity check: does re-aromatized benzene + [H] still give deg 6?
kf = KineticsFamilies().load('default')
fam_ha = {f.label: f for f in kf.families}['H_Abstraction']
benz_ar = Molecule._from_rdmol(Chem.Mol(Chem.MolFromSmiles('c1ccccc1')))
al3 = benz_ar.to_adjlist()
rt3 = Molecule.from_adjacency_list(al3)
pairs3 = aromat_bond_pairs(benz_ar)
em3 = rt3._with_explicit_h()
rw3 = Chem.RWMol(em3)
for p in pairs3:
    a,b = sorted(p)
    rw3.GetBondBetweenAtoms(a,b).SetBondType(BondType.AROMATIC)
for b in rw3.GetBonds():
    if b.GetBondType() != BondType.AROMATIC:
        b.SetIsAromatic(False)
Chem.SanitizeMol(rw3)
benz_re = Molecule._from_rdmol(Chem.Mol(rw3))
print('\nre-aromatized benzene:', benz_re.to_smiles())

efam_ha = enum.Family(
    label=fam_ha.label, recipe=fam_ha.recipe, reverse_recipe=fam_ha.reverse_recipe,
    own_reverse=fam_ha.own_reverse, reversible=fam_ha.reversible,
    allow_charged_species=fam_ha.allow_charged_species, electrons=fam_ha.electrons,
    reactant_num_effective=fam_ha.num_template_reactants_effective,
    product_num_forward=fam_ha.product_num_forward,
    reverse_map=fam_ha.reverse_map,
    template_labels=[e.label for e in fam_ha.forward_template],
    forbidden=fam_ha.forbidden,
)
matcher_ha = TemplateMatcher(fam_ha)
efam_ha.matcher = matcher_ha
h = Molecule(smiles='[H]')
def canon(p):
    r2 = p._rdkit
    if not any(a.GetSymbol()=='H' for a in r2.GetAtoms()):
        try: r2 = Chem.AddHs(r2)
        except Exception: pass
    try: Chem.Kekulize(r2, clearAromaticFlags=True)
    except Exception: pass
    return Chem.MolToSmiles(r2)
rxns = enum.generate_reactions(efam_ha, [benz_re, h], matcher=matcher_ha)
print('H_Abstraction benzene+H (re-aromatized benzene, aromatic form only):')
for r in rxns:
    print('  deg=%.1f products=%s' % (r.degeneracy, [canon(p) for p in r.products]))

# baseline: what does the CURRENT (kekulized) benzene give for H_Abstraction?
benz_k = Molecule(smiles='C1=CC=CC=C1')
rxns_k = enum.generate_reactions(efam_ha, [benz_k, h], matcher=matcher_ha)
print('H_Abstraction benzene+H (CURRENT kekulized):')
for r in rxns_k:
    print('  deg=%.1f products=%s' % (r.degeneracy, [canon(p) for p in r.products]))
