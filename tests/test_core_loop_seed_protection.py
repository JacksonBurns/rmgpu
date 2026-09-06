"""job-06/step-07 + gate re-run: seed species are never demoted from the core."""
import pytest
from rmgpu.core.model import Species
from rmgpu.molecule.molecule import Molecule


def test_seed_species_is_protected_from_demotion():
    from rmgpu.core.loop import CoreEdgeLoop, RunContext, LoopConfig
    from rmgpu.core.family import KineticsFamilies
    # Minimal context
    ctx = RunContext(
        databases=None,
        ml=None,
        families=KineticsFamilies(),
        seed_species=[],
        initial_mole_fractions={},
        temperature=1000.0,
        pressure=1e5,
        config=LoopConfig(),
    )
    loop = CoreEdgeLoop(ctx)
    # Create a seed species
    mol = Molecule(smiles="C")
    sp = Species(label="C", molecule=mol)
    sp.is_seed = True
    loop.species_by_key["k"] = sp
    loop.model.core.species.append(sp)
    # Attempt demote: a seed species must stay in the core
    loop._demote("k")
    assert sp in loop.model.core.species


def test_nonseed_species_can_be_demoted():
    from rmgpu.core.loop import CoreEdgeLoop, RunContext, LoopConfig
    from rmgpu.core.family import KineticsFamilies
    ctx = RunContext(
        databases=None,
        ml=None,
        families=KineticsFamilies(),
        seed_species=[],
        initial_mole_fractions={},
        temperature=1000.0,
        pressure=1e5,
        config=LoopConfig(),
    )
    loop = CoreEdgeLoop(ctx)
    mol = Molecule(smiles="C")
    sp = Species(label="C", molecule=mol)  # is_seed defaults to False
    loop.species_by_key["k"] = sp
    loop.model.core.species.append(sp)
    loop._demote("k")
    assert sp not in loop.model.core.species
    assert sp in loop.model.edge.species


def test_is_seed_key_uses_species_flag():
    from rmgpu.core.loop import CoreEdgeLoop, RunContext, LoopConfig
    from rmgpu.core.family import KineticsFamilies
    ctx = RunContext(
        databases=None,
        ml=None,
        families=KineticsFamilies(),
        seed_species=[],
        initial_mole_fractions={},
        temperature=1000.0,
        pressure=1e5,
        config=LoopConfig(),
    )
    loop = CoreEdgeLoop(ctx)
    sp = Species(label="C", molecule=Molecule(smiles="C"))
    loop.species_by_key["k"] = sp
    assert loop._is_seed_key("k") is False
    sp.is_seed = True
    assert loop._is_seed_key("k") is True
    # unknown keys are never seed-protected
    assert loop._is_seed_key("nope") is False
