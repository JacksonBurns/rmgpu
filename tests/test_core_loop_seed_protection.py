
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
    mol = Molecule.from_smiles('C')
    sp = Species(label='C', molecule=mol)
    sp.is_seed = True
    loop.species_by_key['k'] = sp
    loop.model.core.species.append(sp)
    # Attempt demote
    loop._demote('k')
    assert sp in loop.model.core.species
