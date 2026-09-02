"""Probe 11: which generator produces which form (current code)."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import (
    _is_aromatic, _analyze_molecule,
    _generate_allyl_delocalization_resonance_structures,
    _generate_optimal_aromatic_resonance_structures,
    _generate_kekule_structure,
    generate_resonance_structures,
)

mol = Molecule(smiles='C[CH]C1=CC=CC=C1')
rdmol = mol._rdkit
feats = _analyze_molecule(rdmol)
print('input is_aromatic (stored):', feats['is_aromatic'])

print('\n=== allyl only ===')
for i, s in enumerate(_generate_allyl_delocalization_resonance_structures(rdmol)):
    print('  allyl[%d] %s' % (i, s.to_smiles()))
print('=== optimal_aromatic only ===')
for i, s in enumerate(_generate_optimal_aromatic_resonance_structures(rdmol)):
    print('  opt[%d] %s' % (i, s.to_smiles()))
print('=== kekule only ===')
for i, s in enumerate(_generate_kekule_structure(rdmol)):
    print('  kek[%d] %s' % (i, s.to_smiles()))

print('\n=== full generate_resonance_structures (current) ===')
for i, s in enumerate(generate_resonance_structures(mol)):
    print('  [%d] %s' % (i, s.to_smiles()))
