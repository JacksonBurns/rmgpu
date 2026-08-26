"""
Tests for symmetry number calculation.
"""

import pytest
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.symmetry import get_symmetry_number


@pytest.fixture
def molecule_factory():
    """Factory to create molecules from SMILES."""
    def create(smiles):
        return Molecule(smiles=smiles)
    return create


def test_methane_symmetry(molecule_factory):
    """Test methane - should have symmetry number 12."""
    mol = molecule_factory('C')
    sym = get_symmetry_number(mol)
    assert sym == 12


def test_ethane_symmetry(molecule_factory):
    """Test ethane - RMG-Py gives 18 (C2H6: 3 about C-C + H rotations)."""
    mol = molecule_factory('CC')
    sym = get_symmetry_number(mol)
    assert sym == 18


def test_water_symmetry(molecule_factory):
    """Test water - should have symmetry number 2."""
    mol = molecule_factory('O')
    sym = get_symmetry_number(mol)
    assert sym == 2


def test_ethylene_symmetry(molecule_factory):
    """Test ethylene - RMG-Py gives 4 (C2H4: 2 about C=C + bond)."""
    mol = molecule_factory('C=C')
    sym = get_symmetry_number(mol)
    assert sym == 4


def test_benzene_symmetry(molecule_factory):
    """Test benzene - should have symmetry number 12."""
    mol = molecule_factory('c1ccccc1')
    sym = get_symmetry_number(mol)
    assert sym == 12


def test_cyclopropane_symmetry(molecule_factory):
    """Test cyclopropane - should have symmetry number 6."""
    mol = molecule_factory('C1CC1')
    sym = get_symmetry_number(mol)
    assert sym == 6


def test_is_cyclic(molecule_factory):
    """Test is_cyclic method."""
    cyclic = molecule_factory('C1CC1')
    acyclic = molecule_factory('CCC')
    assert cyclic.is_cyclic()
    assert not acyclic.is_cyclic()
