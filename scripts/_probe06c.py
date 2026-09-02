"""Probe 3: bond types of each resonance form before/after round-trip."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import generate_resonance_structures
from rmgpu.core.enumeration import expand_resonance, assign_fresh_ids
from rdkit import Chem

def bond_report(label, mol):
    m = mol._rdkit
    ar = [b.GetBondTypeAsDouble() for b in m.GetBonds() if b.GetIsAromatic()]
    nonar = [b.GetBondTypeAsDouble() for b in m.GetBonds() if not b.GetIsAromatic()]
    n_ar_atoms = sum(1 for a in m.GetAtoms() if a.GetIsAromatic())
    print('  %-8s smiles=%s  aromatic_bondorders=%s nonaromatic=%s aromatic_atoms=%d' % (
        label, mol.to_smiles(), ar, sorted(set(nonar)), n_ar_atoms))
    # to_adjlist bond symbols for the ring
    al = mol.to_adjlist()
    has_B = 'B}' in al.replace(' ', '')
    print('        to_adjlist has benzene-B bond token: %s' % has_B)

smi = 'C[CH]C1=CC=CC=C1'
mol = Molecule(smiles=smi)
print('=== input stored form ===')
bond_report('input', mol)

forms = generate_resonance_structures(mol)
print('\n=== raw resonance forms (pre-roundtrip) ===')
for i, f in enumerate(forms):
    bond_report('form[%d]' % i, f)

species = expand_resonance([mol])
assign_fresh_ids(species)
print('\n=== after assign_fresh_ids (adjlist round-trip) ===')
for i, f in enumerate(species[0]):
    bond_report('form[%d]' % i, f)
