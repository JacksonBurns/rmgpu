"""Probe 14: test re-aromatize-ring approach for the aromatic_rep form."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu/gates')
from rmgpu.molecule.molecule import Molecule
from rmgpu.core.family import KineticsFamilies
from rmgpu.core.template import TemplateMatcher, _slot_groups
from rmgpu.molecule.group import _explicit_graph, match_explicit_graph, parse_group_adjlist_full
from rmgpu.core import enumeration as enum
from rdkit import Chem
from rdkit.Chem import BondType

def re_aromatize_ring(form):
    """Set the first 6-membered ring's bonds to AROMATIC (1.5)."""
    em = form._rdkit
    for ring in em.GetRingInfo().AtomRings():
        if len(ring) == 6:
            rw = Chem.RWMol(em)
            for k in range(6):
                a, b = ring[k], ring[(k + 1) % 6]
                rw.GetBondBetweenAtoms(a, b).SetBondType(BondType.AROMATIC)
            for b in rw.GetBonds():
                if b.GetBondType() == BondType.AROMATIC:
                    continue
                b.SetIsAromatic(False)
            try:
                Chem.SanitizeMol(rw)
            except Exception as e:
                return None
            form._rdkit = Chem.Mol(rw)
            form._smiles = Chem.MolToSmiles(form._rdkit)
            return form
    return None

kf = KineticsFamilies().load('default')
fam_ie = {f.label: f for f in kf.families}['Intra_ene_reaction']
fam_ra = {f.label: f for f in kf.families}['R_Addition_MultipleBond']

# Build the aromatic_rep form (benzylic) and round-trip + re-aromatize
ar = Molecule._from_rdmol(Chem.Mol(Chem.MolFromSmiles('C[CH]c1ccccc1')))
ar._rdkit.SetProp('rmgpu_aromatic_rep', '1')
al = ar.to_adjlist()
rt = Molecule.from_adjacency_list(al)
rt._rdkit.SetProp('rmgpu_aromatic_rep', '1')
ra_form = re_aromatize_ring(rt)
print('benzylic aromatic_rep form after re-aromatize:', ra_form.to_smiles() if ra_form else None)
g = _explicit_graph(ra_form)
bo = set()
for adj in g['adj']:
    for (j, o) in adj:
        bo.add(round(o, 2))
print('  _explicit_graph bond orders:', sorted(bo))

# Intra_ene template group (the forward reactant)
ie_grp = fam_ie.forward_template[0].item
print('\nIntra_ene forward template group:', len(ie_grp.atoms), 'atoms')
ie_match = match_explicit_graph(g, ie_grp)
print('  benzylic aromatic_rep vs Intra_ene [S,D]/[D,T]: %d matchings (expect 0)' % len(ie_match))

# R_Addition R_R group
ra_grp = fam_ra.forward_template[0].item
print('\nR_Addition R_R group:', len(ra_grp.atoms), 'atoms')
ra_match = match_explicit_graph(g, ra_grp)
print('  benzylic aromatic_rep vs R_Addition [D,T,B]: %d matchings' % len(ra_match))

# Now plain benzene aromatic form (for R_Addition)
benz_ar = Molecule._from_rdmol(Chem.Mol(Chem.MolFromSmiles('c1ccccc1')))
benz_ar._rdkit.SetProp('rmgpu_aromatic_rep', '1')
al2 = benz_ar.to_adjlist()
rt2 = Molecule.from_adjacency_list(al2)
rt2._rdkit.SetProp('rmgpu_aromatic_rep', '1')
benz_ra = re_aromatize_ring(rt2)
print('\nbenzene aromatic_rep after re-aromatize:', benz_ra.to_smiles() if benz_ra else None)
g2 = _explicit_graph(benz_ra)
ra2_match = match_explicit_graph(g2, ra_grp)
print('  benzene aromatic_rep vs R_Addition [D,T,B]: %d matchings (RMG says 12)' % len(ra2_match))

# And run the full R_Addition generation on the re-aromatized benzene to check product+deg
print('\n=== full R_Addition generate_reactions on re-aromatized benzene ===')
efam_ra = enum.Family(
    label=fam_ra.label, recipe=fam_ra.recipe, reverse_recipe=fam_ra.reverse_recipe,
    own_reverse=fam_ra.own_reverse, reversible=fam_ra.reversible,
    allow_charged_species=fam_ra.allow_charged_species, electrons=fam_ra.electrons,
    reactant_num_effective=fam_ra.num_template_reactants_effective,
    product_num_forward=fam_ra.product_num_forward,
    reverse_map=fam_ra.reverse_map,
    template_labels=[e.label for e in fam_ra.forward_template],
    forbidden=fam_ra.forbidden,
)
matcher_ra = TemplateMatcher(fam_ra)
efam_ra.matcher = matcher_ra
h = Molecule(smiles='[H]')
rxns = enum.generate_reactions(efam_ra, [benz_ra, h], matcher=matcher_ra)
for r in rxns:
    def canon(p):
        r2 = p._rdkit
        if not any(a.GetSymbol() == 'H' for a in r2.GetAtoms()):
            try: r2 = Chem.AddHs(r2)
            except Exception: pass
        try: Chem.Kekulize(r2, clearAromaticFlags=True)
        except Exception: pass
        return Chem.MolToSmiles(r2)
    print('  deg=%.1f products=%s' % (r.degeneracy, [canon(p) for p in r.products]))
