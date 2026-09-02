"""Tests for the Chemkin writer."""

import os
import re
import tempfile
import pytest

from rmgpu.schemas.mechanism import MechanismArtifact, MechanismCore, SpeciesEntry, ThermoModel, ReactionEntry, RateParams


def make_test_mechanism() -> MechanismArtifact:
    """Create a hand-built test mechanism with 2 species and 2 reactions."""
    species = [
        SpeciesEntry(
            label="H2O",
            formula="H2O",
            smiles="O",
            adjlist="[O]",
            thermo=ThermoModel(model="wilhoit", Hf298=-241.8, S298=188.8),
        ),
        SpeciesEntry(
            label="O2",
            formula="O2",
            smiles="O=O",
            adjlist="[O]=[O]",
            thermo=ThermoModel(model="wilhoit", Hf298=0.0, S298=205.1),
        ),
    ]
    
    reactions = [
        ReactionEntry(
            label="rxn1",
            reactants=["H2O"],
            products=["O2"],
            family="R_Reactions",
            rate=RateParams(
                type="arrhenius",
                params={"A": 1e10, "n": 0.0, "Ea": 50000.0},
            ),
        ),
        ReactionEntry(
            label="rxn2",
            reactants=["H2O", "O2"],
            products=["H2O", "O2"],
            family="R_Reactions",
            rate=RateParams(
                type="falloff",
                params={
                    "A": 1e12,
                    "n": 0.0,
                    "Ea": 30000.0,
                    "A_low": 1e14,
                    "n_low": 0.0,
                    "Ea_low": 20000.0,
                },
            ),
        ),
    ]
    
    core = MechanismCore(species=species, reactions=reactions)
    return MechanismArtifact(core=core)


def test_write_chemkin_creates_files():
    """Test that write_chemkin creates all three output files."""
    mechanism = make_test_mechanism()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from rmgpu.io.chemkin import write_chemkin
        
        write_chemkin(mechanism, tmpdir)
        
        # Check that all three files exist
        assert os.path.exists(os.path.join(tmpdir, "chem.inp"))
        assert os.path.exists(os.path.join(tmpdir, "chem_annotated.inp"))
        assert os.path.exists(os.path.join(tmpdir, "species_dictionary.txt"))


def test_chem_in_species_count():
    """Test that chem.inp has the correct number of species."""
    mechanism = make_test_mechanism()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from rmgpu.io.chemkin import write_chemkin
        
        write_chemkin(mechanism, tmpdir)
        
        with open(os.path.join(tmpdir, "chem.inp")) as f:
            content = f.read()
        
        # Count species lines - lines that have a species name followed by G phase marker
        # The format is: name(16 chars) + comment(6 chars) + elements(30 chars) + G + ...
        # G is at position 52 (0-indexed)
        lines = content.splitlines()
        # Count lines with 'G' in SPECIES section
        species_count = 0
        in_species = False
        for line in lines:
            if line.strip() == "SPECIES":
                in_species = True
                continue
            if in_species and line.strip() == "END":
                break
            if in_species and len(line) > 52 and line[52].strip().upper() == 'G':
                species_count += 1
        assert species_count == 2, f"Expected 2 species, got {species_count}"


def test_chem_in_reaction_count():
    """Test that chem.inp has the correct number of reactions."""
    mechanism = make_test_mechanism()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from rmgpu.io.chemkin import write_chemkin
        
        write_chemkin(mechanism, tmpdir)
        
        with open(os.path.join(tmpdir, "chem.inp")) as f:
            content = f.read()
        
        # Count reaction lines - they contain "=" and end with rate parameters
        lines = content.splitlines()
        reaction_lines = []
        in_reactions = False
        for line in lines:
            if "REACTIONS" in line:
                in_reactions = True
                continue
            if in_reactions and line.startswith("END"):
                in_reactions = False
                continue
            if in_reactions and "=" in line and not line.startswith("!"):
                reaction_lines.append(line)
        
        assert len(reaction_lines) == 2, f"Expected 2 reactions, got {len(reaction_lines)}"


def test_chem_in_rate_params_arrhenius():
    """Test that Arrhenius rate parameters are correctly written."""
    mechanism = make_test_mechanism()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from rmgpu.io.chemkin import write_chemkin
        
        write_chemkin(mechanism, tmpdir)
        
        with open(os.path.join(tmpdir, "chem.inp")) as f:
            content = f.read()
        
        # Find the first reaction (Arrhenius)
        lines = content.splitlines()
        reaction_lines = []
        in_reactions = False
        for line in lines:
            if "REACTIONS" in line:
                in_reactions = True
                continue
            if in_reactions and line.startswith("END"):
                break
            if in_reactions and "=" in line and not line.startswith("!"):
                reaction_lines.append(line)
        
        # Check first reaction has Arrhenius params
        if reaction_lines:
            rxn_line = reaction_lines[0]
            # Extract the rate parameters (last 3 numbers)
            parts = rxn_line.split()
            # The format is: reaction_string A n Ea
            # Find the numbers at the end
            rate_params = []
            for part in reversed(parts):
                try:
                    rate_params.append(float(part))
                    if len(rate_params) == 3:
                        break
                except ValueError:
                    continue
            
            if len(rate_params) == 3:
                Ea, n, A = rate_params[0], rate_params[1], rate_params[2]
                # Ea should be 50000 J/mol = 50000/4184 kcal/mol
                assert abs(Ea - 50000.0 / 4184.0) < 1.0, f"Ea mismatch: {Ea}"
                assert abs(n - 0.0) < 0.1, f"n mismatch: {n}"
                assert abs(A - 1e10) < 1e9, f"A mismatch: {A}"


def test_species_dictionary():
    """Test that species_dictionary.txt contains all species."""
    mechanism = make_test_mechanism()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from rmgpu.io.chemkin import write_chemkin
        
        write_chemkin(mechanism, tmpdir)
        
        with open(os.path.join(tmpdir, "species_dictionary.txt")) as f:
            content = f.read()
        
        # Check that both species are present
        assert "H2O" in content
        assert "O2" in content
        assert "SMILES" in content


def test_chem_in_falloff_line():
    """Test that falloff reactions have LOW and EFF lines."""
    mechanism = make_test_mechanism()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from rmgpu.io.chemkin import write_chemkin
        
        write_chemkin(mechanism, tmpdir)
        
        with open(os.path.join(tmpdir, "chem.inp")) as f:
            content = f.read()
        
        # Check for LOW/ line
        assert "LOW/" in content, "Missing LOW/ line for falloff reaction"


def test_chem_in_elements():
    """Test that the ELEMENTS section is present."""
    mechanism = make_test_mechanism()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from rmgpu.io.chemkin import write_chemkin
        
        write_chemkin(mechanism, tmpdir)
        
        with open(os.path.join(tmpdir, "chem.inp")) as f:
            content = f.read()
        
        assert "ELEMENTS" in content
        assert "END" in content
        # Check that elements H, O are present
        assert "H" in content
        assert "O" in content
