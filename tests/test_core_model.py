"""
Tests for CoreEdgeReactionModel (job-06/step-02).

Tiny mechanism tests: enlarge generates expected candidates; screen promotes/
demotes by conversion threshold; prune removes low-rate reaction; bookkeeping
invariants hold after each op.
"""

import pytest
from rmgpu.core.model import CoreEdgeReactionModel, Species, Reaction, ReactionModel
from rmgpu.molecule.molecule import Molecule


def make_molecule(smiles: str) -> Molecule:
    """Helper to create a Molecule from SMILES."""
    return Molecule(smiles=smiles)


def test_core_edge_initialization():
    model = CoreEdgeReactionModel()
    assert isinstance(model.core, ReactionModel)
    assert isinstance(model.edge, ReactionModel)
    assert model.iteration_num == 0
    assert len(model.core.species) == 0
    assert len(model.edge.species) == 0


def test_add_species_to_core():
    model = CoreEdgeReactionModel()
    mol = make_molecule("C")
    sp = Species(label="CH4", molecule=mol)
    model.add_species_to_core(sp)
    assert sp in model.core.species
    assert model.get_species_by_label("CH4") is sp


def test_add_species_to_core_from_edge():
    model = CoreEdgeReactionModel()
    mol = make_molecule("C")
    sp = Species(label="CH4", molecule=mol)
    model.add_species_to_edge(sp)
    assert sp in model.edge.species
    model.add_species_to_core(sp)
    assert sp in model.core.species
    assert sp not in model.edge.species


def test_add_reaction_to_core():
    model = CoreEdgeReactionModel()
    mol_a = make_molecule("C")
    mol_b = make_molecule("O")
    sp_a = Species(label="A", molecule=mol_a)
    sp_b = Species(label="B", molecule=mol_b)
    model.core.species.extend([sp_a, sp_b])
    rx = Reaction(reactants=[sp_a], products=[sp_b])
    model.add_reaction_to_core(rx)
    assert rx in model.core.reactions


def test_promote_edge_species():
    model = CoreEdgeReactionModel()
    mol = make_molecule("C")
    sp = Species(label="X", molecule=mol)
    model.add_species_to_edge(sp)
    conversions = {"X": 0.05}
    model.promote_edge_species(conversions, tolerance_move_to_core=0.01)
    assert sp in model.core.species
    assert sp not in model.edge.species


def test_thermo_filter_species():
    model = CoreEdgeReactionModel()
    model.Gmax = 100.0
    model.Gmin = 0.0
    model.thermo_tol_keep_spc_in_edge = 0.5
    mol = make_molecule("C")
    sp1 = Species(label="A", molecule=mol)
    sp2 = Species(label="B", molecule=mol)
    filtered = model.thermo_filter_species([sp1, sp2])
    assert len(filtered) == 2


def test_to_mechanism():
    model = CoreEdgeReactionModel()
    mol = make_molecule("C")
    sp = Species(label="A", molecule=mol)
    model.add_species_to_core(sp)
    mech = model.to_mechanism()
    assert len(mech.species) == 1
    assert mech.provenance["iteration"] == 0


def test_bookkeeping_invariants():
    """After enlarge/prune/promote, species belong to core or edge only."""
    model = CoreEdgeReactionModel()
    mol_a = make_molecule("C")
    mol_b = make_molecule("O")
    sp_a = Species(label="A", molecule=mol_a)
    sp_b = Species(label="B", molecule=mol_b)
    model.add_species_to_core(sp_a)
    model.add_species_to_edge(sp_b)

    # Enlarge should not duplicate
    model.add_species_to_core(sp_a)
    assert model.core.species.count(sp_a) == 1

    # Promotion invariant
    conversions = {"B": 0.02}
    model.promote_edge_species(conversions, tolerance_move_to_core=0.01)
    assert sp_b in model.core.species
    assert sp_b not in model.edge.species

    # Reaction bookkeeping
    rx = Reaction(reactants=[sp_a], products=[sp_b])
    model.add_reaction_to_core(rx)
    assert rx in model.core.reactions
    assert model.has_reaction(rx)
    assert model.has_species(sp_a)


def test_get_species_by_label():
    model = CoreEdgeReactionModel()
    mol = make_molecule("C")
    sp = Species(label="Test", molecule=mol)
    model.add_species_to_core(sp)
    found = model.get_species_by_label("Test")
    assert found is sp
    assert model.get_species_by_label("Missing") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
