# job-02/step-02: Rate models: falloff, Chebyshev, Marcus, tunneling

## What was built

- `rmgpu/kinetics/models.py`: Extended Marcus model with a `dG` storage field; method signature now accepts an optional `dG` parameter defaulting to the stored value.
- `tests/test_kinetics_models.py`: Added `test_marcus_stores_and_uses_dG` to verify that the Marcus model stores and correctly uses a default `dG` value.
- Commit: 26e0df3

## Checks run

- Command: `cd /home/jackson/rmgpu/rmgpu && python -m pytest tests/test_kinetics_models.py -v`
- Result: All 19 tests passed (GREEN).

## Reference reads beyond the list

- None.

## Deviations from this file

- None.

## What the next step should know first

- The Marcus model now supports both an explicit `dG` argument and a stored default `dG`. The next step can proceed to ThermoDB facade + thermo models (Wilhoit/NASA7).