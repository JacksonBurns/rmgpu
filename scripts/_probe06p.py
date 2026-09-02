"""Probe 16: verify new generate_resonance_structures -> 5 forms + flags."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import (generate_resonance_structures,
                                      _form_is_aromatic, _has_standard_kekule_ring)
from rdkit import Chem
from rdkit.Chem import BondType

def flags(m):
    r = m._rdkit
    arom = r.HasProp('rmgpu_aromatic_rep') and r.GetProp('rmgpu_aromatic_rep') == '1'
    rea = r.GetProp('rmgpu_reactive') if r.HasProp('rmgpu_reactive') else '?'
    return 'arom_rep=%s reactive=%s' % (arom, rea)

for smi, name in [('C[CH]C1=CC=CC=C1', 'benzylic(1-phenylethyl)'),
                  ('c1ccccc1', 'benzene'),
                  ('C1=CC=CC=C1', 'benzene(kek)')]:
    mol = Molecule(smiles=smi)
    forms = generate_resonance_structures(mol)
    print('=== %s (%s): %d forms ===' % (smi, name, len(forms)))
    for i, f in enumerate(forms):
        print('  [%d] %-24s  %s  is_arom=%s std_kek=%s' % (
            i, f.to_smiles(), flags(f), _form_is_aromatic(f), _has_standard_kekule_ring(f)))

# The exact target: Intra_ene benzylic must give aromatic + ortho x2 + para + kekulized
mol = Molecule(smiles='C[CH]C1=CC=CC=C1')
forms = generate_resonance_structures(mol)
smis = sorted(f.to_smiles() for f in forms)
print('\nbenzylic form SMILES set:', smis)
# RMG target: C[CH]c1ccccc1, CC=C1[CH]C=CC=C1 (x2 ortho), CC=C1C=C[CH]C=C1 (para), C[CH]C1=CC=CC=C1 (kek)
