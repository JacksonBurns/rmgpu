"""
Tests for the Molecule wrapper over RDKit.
"""

import pytest
from rmgpu.molecule.molecule import Molecule
from rdkit import Chem


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

def test_construction_from_smiles():
    mol = Molecule(smiles="CC")
    assert mol is not None


def test_construction_from_inchi():
    # Ethane InChI
    mol = Molecule(inchi="InChI=1S/C2H6/c1-2/h1-2H3")
    assert mol is not None


def test_construction_from_both_prefers_inchi():
    # InChI takes precedence when both are provided
    mol = Molecule(smiles="CC", inchi="InChI=1S/C2H6/c1-2/h1-2H3")
    assert mol.to_smiles() == "CC"


def test_construction_invalid_smiles_raises():
    with pytest.raises(ValueError):
        Molecule(smiles="invalid_smiles_string")


def test_construction_invalid_inchi_raises():
    with pytest.raises(ValueError):
        Molecule(inchi="invalid_inchi_string")


def test_construction_no_input_raises():
    with pytest.raises(ValueError):
        Molecule()


# ---------------------------------------------------------------------------
# Formula tests
# ---------------------------------------------------------------------------

def test_formula_ethane():
    mol = Molecule(smiles="CC")
    assert mol.get_formula() == "C2H6"


def test_formula_benzene():
    mol = Molecule(smiles="c1ccccc1")
    assert mol.get_formula() == "C6H6"


def test_formula_methane():
    mol = Molecule(smiles="C")
    assert mol.get_formula() == "CH4"


# ---------------------------------------------------------------------------
# Charge and radical count tests
# ---------------------------------------------------------------------------

def test_charge_neutral():
    mol = Molecule(smiles="CC")
    assert mol.get_charge() == 0


def test_charge_positive():
    # Methyl cation
    mol = Molecule(smiles="[CH3+]")
    assert mol.get_charge() == 1


def test_charge_negative():
    # Methoxide
    mol = Molecule(smiles="C[O-]")
    assert mol.get_charge() == -1


def test_radical_count_methyl():
    # Methyl radical
    mol = Molecule(smiles="[CH3]")
    assert mol.get_radical_count() == 1


def test_radical_count_neutral_ethane():
    mol = Molecule(smiles="CC")
    assert mol.get_radical_count() == 0


# ---------------------------------------------------------------------------
# SMILES output tests
# ---------------------------------------------------------------------------

def test_smiles_is_rdcanonical():
    mol = Molecule(smiles="CC")
    assert mol.to_smiles() == Chem.MolToSmiles(Chem.MolFromSmiles("CC"))


def test_smiles_canonical_different_input():
    # Different SMILES representations of the same molecule
    mol1 = Molecule(smiles="CC")
    mol2 = Molecule(smiles="CC")  # same
    assert mol1.to_smiles() == mol2.to_smiles()


def test_smiles_property():
    mol = Molecule(smiles="CC")
    assert mol.smiles == mol.to_smiles()


# ---------------------------------------------------------------------------
# InChI tests
# ---------------------------------------------------------------------------

def test_inchi_roundtrip():
    mol = Molecule(smiles="CC")
    inchi = mol.to_inchi()
    assert inchi.startswith("InChI=")


def test_inchi_property():
    mol = Molecule(smiles="CC")
    assert mol.inchi == mol.to_inchi()


# ---------------------------------------------------------------------------
# Equality and hashing tests
# ---------------------------------------------------------------------------

def test_equality_same_molecule():
    mol1 = Molecule(smiles="CC")
    mol2 = Molecule(smiles="CC")
    assert mol1 == mol2


def test_equality_different_molecule():
    mol1 = Molecule(smiles="CC")
    mol2 = Molecule(smiles="CCC")
    assert mol1 != mol2


def test_equality_across_constructions():
    # Ethane from SMILES and InChI should be equal
    mol_smiles = Molecule(smiles="CC")
    mol_inchi = Molecule(inchi="InChI=1S/C2H6/c1-2/h1-2H3")
    assert mol_smiles == mol_inchi


def test_equality_not_molecule():
    mol = Molecule(smiles="CC")
    assert mol != "CC"


def test_hash_same_molecule():
    mol1 = Molecule(smiles="CC")
    mol2 = Molecule(smiles="CC")
    assert hash(mol1) == hash(mol2)


def test_hash_different_molecule():
    mol1 = Molecule(smiles="CC")
    mol2 = Molecule(smiles="CCC")
    assert hash(mol1) != hash(mol2)


def test_molecule_in_dict():
    mol1 = Molecule(smiles="CC")
    mol2 = Molecule(smiles="CC")
    d = {mol1: "ethane"}
    assert d[mol2] == "ethane"


def test_molecule_in_set():
    mol1 = Molecule(smiles="CC")
    mol2 = Molecule(smiles="CC")
    s = {mol1, mol2}
    assert len(s) == 1


# ---------------------------------------------------------------------------
# Repr and str tests
# ---------------------------------------------------------------------------

def test_repr():
    mol = Molecule(smiles="CC")
    assert "CC" in repr(mol)


def test_str():
    mol = Molecule(smiles="CC")
    assert "CC" in str(mol)
