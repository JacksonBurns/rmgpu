"""
Test round-trip stability of adjacency list format on example species.
"""
import pytest
from rmgpu.molecule.molecule import Molecule


def test_ethane_roundtrip_stable():
    """Test that ethane adjlist is string-stable across roundtrips."""
    mol = Molecule(smiles="CC")
    adjlist1 = mol.to_adjlist()
    mol2 = Molecule.from_adjacency_list(adjlist1)
    adjlist2 = mol2.to_adjlist()
    assert adjlist1 == adjlist2, f"Adjlist not stable:\n{adjlist1}\n{adjlist2}"


def test_methane_roundtrip_stable():
    """Test that methane adjlist is string-stable across roundtrips."""
    mol = Molecule(smiles="C")
    adjlist1 = mol.to_adjlist()
    mol2 = Molecule.from_adjacency_list(adjlist1)
    adjlist2 = mol2.to_adjlist()
    assert adjlist1 == adjlist2, f"Adjlist not stable:\n{adjlist1}\n{adjlist2}"


def test_water_roundtrip_stable():
    """Test that water adjlist is string-stable across roundtrips."""
    mol = Molecule(smiles="O")
    adjlist1 = mol.to_adjlist()
    mol2 = Molecule.from_adjacency_list(adjlist1)
    adjlist2 = mol2.to_adjlist()
    assert adjlist1 == adjlist2, f"Adjlist not stable:\n{adjlist1}\n{adjlist2}"


def test_ethylene_roundtrip_stable():
    """Test that ethylene adjlist is string-stable across roundtrips."""
    mol = Molecule(smiles="C=C")
    adjlist1 = mol.to_adjlist()
    mol2 = Molecule.from_adjacency_list(adjlist1)
    adjlist2 = mol2.to_adjlist()
    assert adjlist1 == adjlist2, f"Adjlist not stable:\n{adjlist1}\n{adjlist2}"


def test_benzene_roundtrip_stable():
    """Test that benzene adjlist is string-stable across roundtrips."""
    mol = Molecule(smiles="C1=CC=CC=C1")
    adjlist1 = mol.to_adjlist()
    mol2 = Molecule.from_adjacency_list(adjlist1)
    adjlist2 = mol2.to_adjlist()
    assert adjlist1 == adjlist2, f"Adjlist not stable:\n{adjlist1}\n{adjlist2}"


def test_propane_radical_roundtrip_stable():
    """Test that a radical adjlist is string-stable across roundtrips."""
    mol = Molecule(smiles="[CH2]CC")
    adjlist1 = mol.to_adjlist()
    mol2 = Molecule.from_adjacency_list(adjlist1)
    adjlist2 = mol2.to_adjlist()
    assert adjlist1 == adjlist2, f"Adjlist not stable:\n{adjlist1}\n{adjlist2}"
