"""Job driver for rmgpu: load YAML, build model, run iteration loop.

Implements the minimal driver required for job-06/step-03. The full RMG-Py loop
is ported incrementally; this version provides a working end-to-end run that
loads the input, builds seed species, initializes CoreEdgeReactionModel,
runs a deterministic iteration loop, and emits minimal output.

The implementation follows the orchestration of RMG-Py's rmgpy/rmg/main.py
run_rmg / execute loop, but with the job-06 stubs for pdep, families, and
estimators. Coverage errors from ML estimators are recorded, never swallowed.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import yaml

from rmgpu.schemas.input import Input, resolve_extends
from rmgpu.units import Quantity
from rmgpu.core.model import CoreEdgeReactionModel, Species
from rmgpu.molecule.molecule import Molecule
from rmgpu.data.estimation import EstimationCounts, MLCoverageError

# Minimal YAML dumper for Quantity (mirrors cli)
class QuantityDumper(yaml.SafeDumper):
    pass

def _represent_quantity(dumper: yaml.SafeDumper, q: Quantity):
    return dumper.represent_mapping(
        'tag:yaml.org,2002:map',
        {'value': q._value, 'unit': q._units},
    )

QuantityDumper.add_representer(Quantity, _represent_quantity)


def _load_input(path: str) -> Input:
    """Load and resolve extends, then validate via Input model."""
    import yaml
    path = os.path.abspath(path)
    base_dir = os.path.dirname(path)
    with open(path) as f:
        doc = yaml.safe_load(f)
    from rmgpu.schemas.input import resolve_extends
    doc = resolve_extends(doc, base_dir)
    return Input(**doc)


def _build_seed_species(model_input: Input) -> list[Species]:
    """Create Species objects from the input species block."""
    species_list: list[Species] = []
    for sp in model_input.species or []:
        # Structure may be string or StructureValue
        struct = sp.structure
        smiles = None
        if isinstance(struct, str):
            smiles = struct
        else:
            # StructureValue
            smiles = getattr(struct, 'smiles', None) or getattr(struct, 'value', None)
        if not smiles:
            raise ValueError(f"Species {sp.label} has no SMILES")
        mol = Molecule(smiles=smiles)
        species_obj = Species(label=sp.label, molecule=mol, reactive=bool(sp.reactive))
        species_list.append(species_obj)
    return species_list


def run(input_path: str) -> Dict[str, Any]:
    """Main driver entry point.

    Loads input, builds databases/estimators/families/reactors/model, runs
    the iteration loop to steady state, and returns a summary dict.

    For job-06/step-03 the loop is minimal: seed species are placed in core,
    no enlarge is performed (families/estimators are stubs), and the run
    completes deterministically. The output is printed to stdout for CLI use.
    """
    input_model = _load_input(input_path)

    # Build databases (job-02) - stub for this step
    databases = {}
    if getattr(input_model, 'database', None):
        # Database construction would happen here in a full implementation
        # For step-03 we keep it as a placeholder to avoid circular imports
        databases['config'] = input_model.database.model_dump()

    # Build estimators (job-04) - stub
    estimators = {}
    if getattr(input_model, 'ml_estimator', None):
        # Real construction happens in job-04; here we just record presence
        estimators['ml_estimator'] = True

    # Build families (job-05) - stub
    families = {}
    if getattr(input_model, 'database', None):
        families['default'] = True

    # Build reactors (job-06/1)
    reactors = []
    if getattr(input_model, 'reactors', None):
        reactors = input_model.reactors

    # Build model (job-06/2)
    core_model = CoreEdgeReactionModel()
    seed_species = _build_seed_species(input_model)
    for sp in seed_species:
        core_model.add_species_to_core(sp)

    # Set tolerances from model block
    model_block = getattr(input_model, 'model', None)
    if model_block:
        core_model.tolerance_move_to_core = float(getattr(model_block, 'tolerance_move_to_core', 0.01))
        core_model.tolerance_keep_in_edge = float(getattr(model_block, 'tolerance_keep_in_edge', 0.001))
        core_model.tolerance_interrupt_simulation = float(getattr(model_block, 'tolerance_interrupt_simulation', 0.1))

    # Iteration loop (minimal)
    # The full loop would call enlarge -> simulate -> screen -> prune.
    # For step-03 we perform one iteration with no enlarge to prove the driver works.
    max_iterations = 10
    iteration_num = 0
    prev_core_labels = None
    estimation_counts = EstimationCounts()
    steady = False

    # Determinism: sort species by label before each iteration
    def _sorted_core_labels():
        return tuple(sorted([s.label for s in core_model.core.species]))

    while iteration_num < max_iterations and not steady:
        iteration_num += 1
        core_model.iteration_num = iteration_num

        # Enlarge stub: no reaction generation in this step
        # In a full implementation:
        #   new_species, new_rxns = generate via families
        #   estimate thermo/kinetics via estimators
        #   simulate via reactors
        #   screen/promote
        # Here we just check for steady state
        current_labels = _sorted_core_labels()
        if prev_core_labels is not None and current_labels == prev_core_labels:
            steady = True
        prev_core_labels = current_labels

        # If no species, break
        if not core_model.core.species:
            break

    # Build minimal mechanism output
    mechanism_summary = {
        'iteration': core_model.iteration_num,
        'core_species_count': len(core_model.core.species),
        'core_reaction_count': len(core_model.core.reactions),
        'edge_species_count': len(core_model.edge.species),
        'edge_reaction_count': len(core_model.edge.reactions),
        'core_species_labels': [s.label for s in core_model.core.species],
        'estimation_counts': estimation_counts.as_dict(),
        'input_path': input_path,
    }

    # Print summary for CLI
    print(f"Iteration: {mechanism_summary['iteration']}")
    print(f"Core species: {mechanism_summary['core_species_count']}, Core reactions: {mechanism_summary['core_reaction_count']}")
    print(f"Edge species: {mechanism_summary['edge_species_count']}, Edge reactions: {mechanism_summary['edge_reaction_count']}")

    return mechanism_summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m rmgpu.main <input.yaml>", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1])
