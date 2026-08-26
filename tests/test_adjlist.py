"""
Tests for adjacency list parsing and serialization.
"""
import pytest
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.adjlist import InvalidAdjacencyListError


class TestAdjlistSerialization:
    """Test serialization of molecules to adjacency list format."""

    def test_ethane(self):
        """Test serialization of ethane."""
        mol = Molecule(smiles="CC")
        adjlist = mol.to_adjlist()
        assert "C" in adjlist
        assert "H" in adjlist
        # Should have 2 carbon atoms and 6 hydrogen atoms
        assert adjlist.count("C") >= 2

    def test_ethane_roundtrip(self):
        """Test that serialization and parsing roundtrip works."""
        mol = Molecule(smiles="CC")
        adjlist = mol.to_adjlist()
        mol2 = Molecule.from_adjacency_list(adjlist)
        assert mol.is_isomorph(mol2)

    def test_water(self):
        """Test serialization of water."""
        mol = Molecule(smiles="O")
        adjlist = mol.to_adjlist()
        assert "O" in adjlist
        assert "H" in adjlist
        # Should have 1 oxygen and 2 hydrogens
        assert adjlist.count("O") >= 1

    def test_water_roundtrip(self):
        """Test that water roundtrips correctly."""
        mol = Molecule(smiles="O")
        adjlist = mol.to_adjlist()
        mol2 = Molecule.from_adjacency_list(adjlist)
        assert mol.is_isomorph(mol2)

    def test_ch4(self):
        """Test serialization of methane."""
        mol = Molecule(smiles="C")
        adjlist = mol.to_adjlist()
        assert "C" in adjlist
        assert "H" in adjlist
        assert adjlist.count("H") >= 4

    def test_ch4_roundtrip(self):
        """Test that methane roundtrips correctly."""
        mol = Molecule(smiles="C")
        adjlist = mol.to_adjlist()
        mol2 = Molecule.from_adjacency_list(adjlist)
        assert mol.is_isomorph(mol2)


class TestAdjlistParsing:
    """Test parsing of adjacency list format."""

    def test_parse_simple(self):
        """Test parsing a simple adjacency list."""
        adjlist = """
1  C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2  H u0 p0 c0 {1,S}
3  H u0 p0 c0 {1,S}
4  H u0 p0 c0 {1,S}
5  H u0 p0 c0 {1,S}
"""
        mol = Molecule.from_adjacency_list(adjlist)
        assert mol.get_formula() == "CH4"

    def test_parse_with_multiplicity(self):
        """Test parsing adjacency list with explicit multiplicity."""
        adjlist = """
multiplicity 2
1  C u1 p0 c0 {2,S} {3,S} {4,S}
2  H u0 p0 c0 {1,S}
3  H u0 p0 c0 {1,S}
4  H u0 p0 c0 {1,S}
"""
        mol = Molecule.from_adjacency_list(adjlist)
        assert mol.get_formula() == "CH3"
        assert mol.get_radical_count() == 1

    def test_parse_double_bond(self):
        """Test parsing adjacency list with double bond."""
        adjlist = """
1  C u0 p0 c0 {2,D} {3,S} {4,S}
2  O u0 p2 c0 {1,D}
3  H u0 p0 c0 {1,S}
4  H u0 p0 c0 {1,S}
"""
        mol = Molecule.from_adjacency_list(adjlist)
        assert mol.get_formula() == "CH2O"

    def test_parse_triple_bond(self):
        """Test parsing adjacency list with triple bond."""
        adjlist = """
1  C u0 p0 c0 {2,T} {3,S}
2  N u0 p0 c0 {1,T}
3  H u0 p0 c0 {1,S}
"""
        mol = Molecule.from_adjacency_list(adjlist)
        assert mol.get_formula() == "CHN"

    def test_parse_with_labels(self):
        """Test parsing adjacency list with labels."""
        adjlist = """
1 *1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2    H u0 p0 c0 {1,S}
3    H u0 p0 c0 {1,S}
4    H u0 p0 c0 {1,S}
5 *2 H u0 p0 c0 {1,S}
"""
        mol = Molecule.from_adjacency_list(adjlist)
        assert mol.contains_labeled_atom("*1")
        assert mol.contains_labeled_atom("*2")

    def test_parse_charge(self):
        """Test parsing adjacency list with charges."""
        adjlist = """
1  C u0 p0 c+1 {2,D} {3,S} {4,S}
2  O u0 p0 c-1 {1,D}
3  H u0 p0 c0 {1,S}
4  H u0 p0 c0 {1,S}
"""
        mol = Molecule.from_adjacency_list(adjlist)
        # Total charge should be 0 (+1 and -1)
        assert mol.get_charge() == 0

    def test_parse_isotope(self):
        """Test parsing adjacency list with isotope."""
        adjlist = """
1  C u0 p0 c0 i14 {2,S} {3,S} {4,S} {5,S}
2  H u0 p0 c0 {1,S}
3  H u0 p0 c0 {1,S}
4  H u0 p0 c0 {1,S}
5  H u0 p0 c0 {1,S}
"""
        mol = Molecule.from_adjacency_list(adjlist)
        assert mol.get_formula() == "CH4"


class TestAdjlistErrors:
    """Test error handling in adjacency list parsing."""

    def test_empty_adjlist(self):
        """Test that empty adjacency list raises error."""
        with pytest.raises(InvalidAdjacencyListError):
            Molecule.from_adjacency_list("")

    def test_invalid_element(self):
        """Test that invalid element raises error."""
        adjlist = """
1  X u0 p0 c0 {2,S}
2  H u0 p0 c0 {1,S}
"""
        with pytest.raises(InvalidAdjacencyListError):
            Molecule.from_adjacency_list(adjlist)

    def test_missing_bond_target(self):
        """Test that missing bond target raises error."""
        adjlist = """
1  C u0 p0 c0 {2,S}
"""
        with pytest.raises(InvalidAdjacencyListError):
            Molecule.from_adjacency_list(adjlist)
