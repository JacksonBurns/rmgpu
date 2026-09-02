"""Probe: what does assign_fresh_ids actually leave on each form?"""
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
print('== pre-round-trip (from generate_resonance_structures) ==')
for i, f in enumerate(forms):
    r = f._rdkit
    ar = r.HasProp('rmgpu_aromatic_rep') and r.GetProp('rmgpu_aromatic_rep')=='1'
    rea = r.GetProp('rmgpu_reactive') if r.HasProp('rmgpu_reactive') else 'NO-PROP'
    print('  [%d] ar_rep=%s reactive=%s smiles=%s' % (i, ar, rea, f.to_smiles()))

species = [list(forms)]
enum.assign_fresh_ids(species)
rt = species[0]
print('\n== post-round-trip (after assign_fresh_ids) ==')
for i, f in enumerate(rt):
    r = f._rdkit
    ar = r.HasProp('rmgpu_aromatic_rep') and r.GetProp('rmgpu_aromatic_rep')=='1'
    rea = r.GetProp('rmgpu_reactive') if r.HasProp('rmgpu_reactive') else 'NO-PROP'
    # ring bond orders
    em = f._with_explicit_h()
    ringo = []
    for ring in em.GetRingInfo().AtomRings():
        if len(ring)==6:
            for k in range(6):
                a,b = ring[k], ring[(k+1)%6]
                ringo.append(round(em.GetBondBetweenAtoms(a,b).GetBondTypeAsDouble(),2))
    print('  [%d] ar_rep=%s reactive=%s ring=%s smiles=%s' % (i, ar, rea, sorted(ringo), f.to_smiles()[:30]))

# Now matcher matchings
kf = KineticsFamilies().load('default')
fam = {f.label: f for f in kf.families}['Intra_ene_reaction']
matcher = TemplateMatcher(fam)
print('\n== Intra_ene matchings per post-rt form ==')
for i, f in enumerate(rt):
    ms = matcher.match_molecule(f, 0, 'ab')
    print('  form %d: %d matchings' % (i, len(ms)))
