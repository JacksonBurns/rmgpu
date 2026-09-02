"""Probe 5: per-allyl-shift for benzylic input (what the BFS would generate)."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import (_generate_allyl_delocalization_resonance_structures,
                                      _is_radical)
from rdkit import Chem

def key(rdmol):
    ak = tuple((a.GetIdx(), a.GetSymbol(), a.GetNumRadicalElectrons(), a.GetFormalCharge())
               for a in rdmol.GetAtoms())
    bk = tuple((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                round(b.GetBondTypeAsDouble(), 2), b.GetIsAromatic())
               for b in rdmol.GetBonds())
    return (ak, bk)

mol = Molecule(smiles='C[CH]C1=CC=CC=C1')
print('input to_smiles:', mol.to_smiles())

# one-pass shifts of the input
shifts = _generate_allyl_delocalization_resonance_structures(mol._rdkit)
print('\n=== one-pass allyl shifts of input: %d ===' % len(shifts))
for i, s in enumerate(shifts):
    print('  [%d] %s  key_atoms_rad=%s' % (
        i, s.to_smiles(),
        [a.GetIdx() for a in s._rdkit.GetAtoms() if a.GetNumRadicalElectrons() > 0]))

# Now BFS: from each shift, generate its shifts
print('\n=== BFS: shifts of each shift ===')
seen = {key(mol._rdkit)}
frontier = list(shifts)
gen = []
while frontier:
    f = frontier.pop(0)
    for s in _generate_allyl_delocalization_resonance_structures(f._rdkit):
        k = key(s._rdkit)
        if k not in seen:
            seen.add(k)
            gen.append(s)
            frontier.append(s)
            print('  NEW %s  rad_atoms=%s' % (
                s.to_smiles(),
                [a.GetIdx() for a in s._rdkit.GetAtoms() if a.GetNumRadicalElectrons() > 0]))

print('\nTotal BFS-generated (beyond input):', len(gen))
for i, s in enumerate(gen):
    print('  [%d] %s' % (i, s.to_smiles()))
