"""Compare rmgpu explicit-H vertex element ordering vs RMG from_smiles ordering.
Determines whether (react_atom_idx, tmpl_atom_idx) parity is directly comparable."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rdkit import Chem

def rmgpu_seq(m):
    # explicit-H graph the matcher will use
    em = m._with_explicit_h()
    return [a.GetSymbol() for a in em.GetAtoms()]

# RMG side (run separately) - here we only show rmgpu side + the group side
for s in ['C', 'CC', '[CH3]', 'C1=CC=CC=C1', 'C#C']:
    m = Molecule(smiles=s)
    print('rmgpu %-16s n=%d seq=%s' % (s, len(rmgpu_seq(m)), rmgpu_seq(m)))
