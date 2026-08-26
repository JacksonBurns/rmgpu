"""Tests for atom type assignment."""

import pytest
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.atomtype import assign_atom_types

# Test molecules - need at least 30 diverse examples
TEST_MOLECULES = [
    'C',           # methane
    'CC',          # ethane
    'C=C',         # ethylene
    'C#C',         # acetylene
    'CCO',         # ethanol
    'CC(=O)C',     # acetone
    'c1ccccc1',    # benzene
    'N',           # ammonia
    'N=N',         # azo compound
    'N#N',         # dinitrogen
    'O',           # water
    'O=C=O',       # carbon dioxide
    'C1CCCCC1',    # cyclohexane
    'CCN',         # ethylamine
    'CC(=O)O',     # acetic acid
    'CCSCC',       # 1,2-dimethylsulfide
    'CCC',         # propane
    'C(=O)C',      # formaldehyde
    'CC(F)(F)F',   # fluoroethane
    'CCCl',        # chloroethane
    'CCBr',        # bromoethane
    'CC#N',        # acetonitrile
    'CC(=O)N',     # acetamide
    'CCS',         # methyl sulfide
    'C=C(C)C',     # isobutylene
    'CC(C)(C)C',   # neopentane
    'C1=C2C(=C1)C=CC=C2',  # naphthalene
    'C1=CC=NC=N1',   # pyrazine
    'CCOC',        # diethyl ether
    'CC(C)O',      # isopropanol
]

def test_assign_atom_types_returns_list():
    """Test that atom type assignment returns a list."""
    mol = Molecule(smiles='CC')
    atom_types = assign_atom_types(mol)
    assert isinstance(atom_types, list)

def test_assign_atom_types_length_matches_atoms():
    """Test that atom types list length matches the explicit-H graph (RMG-Py)."""
    mol = Molecule(smiles='CC')
    atom_types = assign_atom_types(mol)
    # rmgpu assigns atom types on the explicit-hydrogen graph (as RMG-Py does)
    n_atoms = mol._with_explicit_h().GetNumAtoms()
    assert len(atom_types) == n_atoms
    assert len(atom_types) == 8  # 2 C + 6 H

def test_carbon_atoms_get_c_type():
    """Test that carbon atoms are assigned the C5s (Cs) RMG-Py type."""
    mol = Molecule(smiles='CC')
    atom_types = assign_atom_types(mol)
    # Non-hydrogen atoms: both carbons are Cs
    non_h = [t for t in atom_types if t != 'H0']
    assert non_h == ['Cs', 'Cs']
    assert all(t.startswith('C') for t in non_h)

def test_oxygen_atoms_get_o_type():
    """Test that the oxygen atom is assigned an O* RMG-Py type."""
    mol = Molecule(smiles='CO')
    atom_types = assign_atom_types(mol)
    o_types = [t for t in atom_types if t.startswith('O')]
    assert o_types == ['O2s']

def test_nitrogen_atoms_get_n_type():
    """Test that the nitrogen atom is assigned an N* RMG-Py type."""
    mol = Molecule(smiles='CN')
    atom_types = assign_atom_types(mol)
    n_types = [t for t in atom_types if t.startswith('N')]
    assert n_types == ['N3s']

def test_all_molecules_have_atom_types():
    """Test that all molecules can have atom types assigned (explicit-H graph)."""
    for smiles in TEST_MOLECULES:
        mol = Molecule(smiles=smiles)
        atom_types = assign_atom_types(mol)
        n_atoms = mol._with_explicit_h().GetNumAtoms()
        assert len(atom_types) == n_atoms, f"Failed for {smiles}"

def test_atom_types_are_strings():
    """Test that all atom types are strings."""
    mol = Molecule(smiles='CCCO')
    atom_types = assign_atom_types(mol)
    for atom_type in atom_types:
        assert isinstance(atom_type, str)

def test_benzene_carbons():
    """Test benzene ring carbon assignment (explicit-H graph)."""
    mol = Molecule(smiles='c1ccccc1')
    atom_types = assign_atom_types(mol)
    # Non-hydrogen atoms in benzene: all six carbons are Cd (C in a ring)
    non_h = [t for t in atom_types if t != 'H0']
    assert non_h == ['Cd'] * 6

def test_charged_atoms():
    """Test atom type assignment for charged atoms."""
    mol = Molecule(smiles='[C+]')
    atom_types = assign_atom_types(mol)
    assert len(atom_types) == 1

def test_radical_atoms():
    """Test atom type assignment for radical atoms."""
    mol = Molecule(smiles='[C]')
    atom_types = assign_atom_types(mol)
    assert len(atom_types) == 1

@pytest.mark.parametrize("smiles", TEST_MOLECULES[:10])
def test_individual_molecules(smiles):
    """Test individual molecules (explicit-H graph)."""
    mol = Molecule(smiles=smiles)
    atom_types = assign_atom_types(mol)
    n_atoms = mol._with_explicit_h().GetNumAtoms()
    assert len(atom_types) == n_atoms