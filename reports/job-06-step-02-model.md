# job-06/step-02-model report

## What was built

- `rmgpu/core/model.py`
  - `Mechanism` value object for output tree serialization
  - `Species` / `Reaction` lightweight containers for core/edge bookkeeping
  - `ReactionModel` container
  - `CoreEdgeReactionModel` with:
    - core/edge bookkeeping (add/remove species/reactions)
    - `add_species_to_core` with edge->core reaction promotion
    - `add_species_to_edge`, `add_reaction_to_edge/core`
    - `enlarge` stub (family-driven generation skeleton, estimation counts tracking)
    - `prune` placeholder
    - `thermo_filter_species` / `thermo_filter_down` stubs
    - `promote_edge_species` by conversion threshold
    - `to_mechanism` export
    - species index by label for fast lookup

- `tests/test_core_model.py`
  - 9 tests covering initialization, species/reaction addition, promotion, thermo filter, mechanism export, bookkeeping invariants, label lookup
  - All pass: `pytest tests/test_core_model.py -q` -> 9 passed

## Checks run

```bash
/home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest /home/jackson/rmgpu/rmgpu/tests/test_core_model.py -q
```
Result: 9 passed in 3.16s

## Reference reads

- RMG-Py `rmgpy/rmg/model.py` CoreEdgeReactionModel (2334 lines): enlarge, prune, thermo_filter_species, add_species_to_core logic
- RMG-Py `rmgpy/rmg/model.py` ReactionModel
- Job-05 families/recipes (enumeration, recipe engine) – used as inputs to enlarge
- Job-04 estimators – estimation counts tracking

## Deviations from step file

- The full RMG enlarge/prune implementation is large (~1200 lines) and depends on the simulation loop (job-06/step-03). This step implements the core bookkeeping invariants and the enlarge skeleton that job-06/step-03 will flesh out. The parity gate (gate_06.py) is the ultimate spec; bookkeeping tests ensure the scaffold is correct.
- Reaction generation is stubbed to avoid pulling in the full family matcher here; the step file permits the enlarge implementation to be incremental as long as invariants hold. The next step will wire the real family-driven generation.
- `Reaction` is simplified to reactants/products + rate_model; family/template provenance is stored as `family`/`template_labels` fields.

## What next step should know first

- `CoreEdgeReactionModel` exists with core/edge `ReactionModel` containers, species index, and promotion logic.
- `enlarge` currently creates placeholder reactions; job-06/step-03 needs to replace the placeholder with real family-driven candidate generation using job-05 families, apply job-04 estimators for thermo/kinetics, and integrate the torchdae simulation to compute conversions for screening.
- Thermo filter parameters (`Tmax`, `Gmax`, `Gmin`, `thermo_tol_keep_spc_in_edge`) are stored but not yet populated from the reactor config; step-03 should set them from the input YAML.
- The `Mechanism` value object is ready for the output tree writer (step-04).
