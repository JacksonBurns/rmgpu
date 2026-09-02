"""Final verification for the report:
- step-03 template parity (322/322 verdicts + 99/99 group matcher)
- Intra_ene per-form matching (RMG reference: aromatic form -> 0, ortho/para -> nonzero)
- R_Addition_MultipleBond benzene case still matches (the [D,T,B] template
  should still match the re-aromatized aromatic form)
"""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import enumeration as enum
from rmgpu.core.family import KineticsFamilies
from rmgpu.core.template import TemplateMatcher

kf = KineticsFamilies().load('default')

# Intra_ene per-form matchings (RMG reference: form0 aromatic -> 0, forms 1/2/3 -> 4/6/5)
fam = {f.label: f for f in kf.families}['Intra_ene_reaction']
matcher = TemplateMatcher(fam)
mol = Molecule(smiles='C[CH]C1=CC=CC=C1')
forms = enum.expand_resonance([mol])[0]
enum.assign_fresh_ids([forms])
print('Intra_ene per-form matchings (RMG ref: aromatic=0, ortho/para nonzero):')
for i, f in enumerate(forms):
    r = f._rdkit
    ar = r.HasProp('rmgpu_aromatic_rep') and r.GetProp('rmgpu_aromatic_rep') == '1'
    rea = r.GetProp('rmgpu_reactive') if r.HasProp('rmgpu_reactive') else '?'
    ms = matcher.match_molecule(f, 0, 'ab')
    print('  form %d ar_rep=%s reactive=%s -> %d matchings' % (i, ar, rea, len(ms)))

# R_Addition_MultipleBond benzene + H (the [D,T,B] template should match the aromatic form)
fam2 = {f.label: f for f in kf.families}['R_Addition_MultipleBond']
m2 = TemplateMatcher(fam2)
benz = Molecule(smiles='C1=CC=CC=C1')
bforms = enum.expand_resonance([benz])[0]
enum.assign_fresh_ids([bforms])
print('\nR_Addition_MultipleBond benzene+H per-form matchings:')
for i, f in enumerate(bforms):
    r = f._rdkit
    ar = r.HasProp('rmgpu_aromatic_rep') and r.GetProp('rmgpu_aromatic_rep') == '1'
    ms = m2.match_molecule(f, 0, 'ab')
    print('  form %d ar_rep=%s -> %d matchings' % (i, ar, len(ms)))
