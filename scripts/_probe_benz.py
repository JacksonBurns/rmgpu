"""Diagnose the benzene/toluene gate_01 regression.
Check: what does each generated form's to_smiles() return, what is
MOLECULE_LOOKUPS for C6H6/C7H8, and what does the clean tree produce?
"""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import generate_resonance_structures, _form_is_aromatic, _has_standard_kekule_ring
import rmgpu.molecule.molecule as MM

print('MOLECULE_LOOKUPS C6H6:', MM.MOLECULE_LOOKUPS.get('C6H6'))
print('MOLECULE_LOOKUPS C7H8:', MM.MOLECULE_LOOKUPS.get('C7H8'))
print()
for smi in ['c1ccccc1', 'Cc1ccccc1']:
    m = Molecule(smiles=smi)
    forms = generate_resonance_structures(m)
    print('===', smi, '->', len(forms), 'forms')
    for i, f in enumerate(forms):
        raw = f._smiles  # the stored canonical smiles
        print('  [%d] to_smiles()=%-16s stored=%-16s is_arom=%s std_kek=%s' % (
            i, f.to_smiles(), raw, _form_is_aromatic(f), _has_standard_kekule_ring(f)))
    print('  reference: benzene=[c1ccccc1, C1=CC=CC=C1] toluene=[Cc1ccccc1, CC1=CC=CC=C1]')
