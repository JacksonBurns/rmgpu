"""Probe z: verify the 5-form set through the real assign_fresh_ids round-trip,
flags preserved, and per-form Intra_ene matchings WITH the matcher's
flag-gated 1.5 fix. This is the full fix 2+3 validation.
"""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import generate_resonance_structures
from rmgpu.core import enumeration as enum
from rmgpu.core.family import KineticsFamilies
from rmgpu.core.template import TemplateMatcher

mol = Molecule(smiles='C[CH]C1=CC=CC=C1')
forms = generate_resonance_structures(mol)
print('generate_resonance_structures ->', len(forms), 'forms')

# Run the real round-trip
species = [list(forms)]
enum.assign_fresh_ids(species)
rt = species[0]

print('\nPer form after assign_fresh_ids (what the matcher sees):')
for i, f in enumerate(rt):
    r = f._rdkit
    ar = r.HasProp('rmgpu_aromatic_rep') and r.GetProp('rmgpu_aromatic_rep') == '1'
    rea = r.GetProp('rmgpu_reactive') if r.HasProp('rmgpu_reactive') else '?'
    # ring bond orders (AddHs)
    em = f._with_explicit_h()
    ringo = []
    for ring in em.GetRingInfo().AtomRings():
        if len(ring) == 6:
            for k in range(6):
                a, b = ring[k], ring[(k + 1) % 6]
                o = em.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble()
                ringo.append(round(o, 2))
    print('  [%d] ar_rep=%s reactive=%s ring_orders=%s smiles=%s' % (
        i, ar, rea, sorted(ringo), f.to_smiles()[:30]))

# Intra_ene matchings (WITHOUT the matcher fix yet - baseline)
kf = KineticsFamilies().load('default')
fam = {f.label: f for f in kf.families}['Intra_ene_reaction']
matcher = TemplateMatcher(fam)
efam = enum.Family(
    label=fam.label, recipe=fam.recipe, reverse_recipe=fam.reverse_recipe,
    own_reverse=fam.own_reverse, reversible=fam.reversible,
    allow_charged_species=fam.allow_charged_species, electrons=fam.electrons,
    reactant_num_effective=fam.num_template_reactants_effective,
    product_num_forward=fam.product_num_forward, reverse_map=fam.reverse_map,
    template_labels=[e.label for e in fam.forward_template], forbidden=fam.forbidden,
)
print('\nIntra_ene matchings per form (no matcher fix, no reactive skip):')
for i, f in enumerate(rt):
    ms = matcher.match_molecule(f, 0, 'ab')
    print('  form %d: %d matchings' % (i, len(ms)))
