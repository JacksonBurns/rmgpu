"""Probe r: diagnose why draft generate_resonance_structures returns 1 form.
Questions:
  A. Is the input benzylic mol is_aromatic (flag-based)?
  B. Does Chem.SetAromaticity aromatize a kekulized benzene ring?
  C. Stage-by-stage: what new_structures / all_structures / filter produce.
  D. Does filter_structures' SMILES dedup collapse the two ortho forms?
"""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
import rmgpu.molecule.resonance as R

mol = Molecule(smiles='C[CH]C1=CC=CC=C1')
rdmol = mol._rdkit
print('A. input is_aromatic (flag):', R._is_aromatic(rdmol))
print('A. input _form_is_aromatic:', R._form_is_aromatic(mol))

# B. SetAromaticity on kekulized benzene
benz = Chem.RWMol(Chem.Mol(Chem.MolFromSmiles('C1=CC=CC=C1')))
try:
    Chem.SetAromaticity(benz)
    ar = [ (b.GetBeginAtomIdx(),b.GetEndAtomIdx(),b.GetBondTypeAsDouble(),b.GetIsAromatic())
           for b in benz.GetBonds() if b.GetBondTypeAsDouble() in (1,2) ]
    print('B. benzene after SetAromaticity (S/D bonds):', ar)
    print('   is_aromatic now:', R._is_aromatic(benz))
    m = Molecule._from_rdmol(Chem.Mol(benz))
    print('   -> Molecule smiles:', m.to_smiles(), 'is_arom:', R._form_is_aromatic(m))
except Exception as e:
    print('B. SetAromaticity failed:', e)

# C. stage-by-stage on the benzylic input
features = R._analyze_molecule(rdmol)
print('\nC. features:', {k: v for k, v in features.items()})
new_structures = []
if features['is_radical']:
    seed = mol.copy()
    if features['is_aromatic']:
        opt = R._generate_optimal_aromatic_resonance_structures(rdmol)
        if opt:
            seed = opt[0]
    print('C. seed is_aromatic:', features['is_aromatic'], 'seed smiles:', seed.to_smiles())
    new_structures.extend(R._allyl_bfs([seed]))
if features['hasLonePairs'] and not features['is_aromatic']:
    new_structures.extend(R._generate_lone_pair_multiple_bond_resonance_structures(rdmol))
    new_structures.extend(R._generate_adj_lone_pair_radical_resonance_structures(rdmol))
if features['is_aromatic']:
    new_structures.extend(R._generate_optimal_aromatic_resonance_structures(rdmol))
    new_structures.extend(R._generate_kekule_structure(rdmol))
print('C. new_structures count:', len(new_structures))
for f in new_structures:
    print('   ', f.to_smiles())
all_structures = [mol.copy()]
seen = {R._form_key(all_structures[0])}
for s in new_structures:
    k = R._form_key(s)
    if k not in seen:
        seen.add(k); all_structures.append(s)
print('C. all_structures (key-dedup) count:', len(all_structures))
for f in all_structures:
    print('   ', f.to_smiles())
from rmgpu.molecule.resonance_filtration import filter_structures
res = filter_structures(all_structures, mol.copy())
print('C. filter_structures count:', len(res))
for f in res:
    print('   ', f.to_smiles())

# D. SMILES dedup collapse test: two isomorphic ortho forms
o1 = Molecule(smiles='CC=C1[CH]C=CC=C1')
# build the isomer with radical at the other ortho by shifting
print('\nD. ortho form smiles:', o1.to_smiles())
