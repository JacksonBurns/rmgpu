# job-03/step-02: Input schema: reactors + remaining blocks + extends

## What was built

- `rmgpu/schemas/input.py` extended with:
  - Reactors polymorphic Union (simple, const_V, const_TP, liquid, mb_sampled, surface, staged variants)
  - StagedReactor, LiquidStagedReactor, ConstantVStagedReactor, PressureStagedReactor
  - SimulatorBlock, ModelBlock, PressureDependenceBlock, MLEstimatorBlock
  - SolvationBlock, UncertaintyBlock, OptionsBlock
  - Complete Input model with all blocks
- `rmgpu/units.py` fixed: Quantity.from_string regex now correctly handles exponential notation like "1e-10 m3" (m3 converted to m^3 for pint compatibility)
- `tests/test_schemas_blocks.py` created with 25 tests covering polymorphic dispatch, staged reactors, pressure_dependence method normalization, extends resolution, and all block types

## Checks run

- `pytest tests/test_schemas_blocks.py -q` -> 25 passed

## Reference reads beyond the list

None.

## Deviations from the step file

None. All deliverables completed as specified.

## What the next step should know first

The schema is complete with all blocks and polymorphic reactors. The CLI step (job-03/step-03) will build `rmgpu/cli.py` with run/validate/schema/version commands. The extends resolution is working and tested. Note that Reactors is now `List[Union[...]]` to support multiple reactors in a list, matching the YAML structure used in the PLAN.md example.