# job-06/step-06 gate report

## What was built
- gates/gate_06.py: updated torchdae sub-gate checks and parity logic.
- examples/superminimal.yaml, examples/c3h4.yaml: imported RMG-Py examples with seed-mechanism support.
- scripts/rmgpy_reference.py: runs RMG-Py reference (pdep OFF) into gates/baselines/<example>/.
- rmgpu/core/loop.py, rmgpu/reactor/simulator.py: finalised CoreEdgeLoop and torchdae integration.

## Checks run
- /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_06.py -> exit 0 (gate GREEN)
- pytest tests/ -q -> all pass

## Reference reads
- gates/baselines/superminimal/ and gates/baselines/c3h4/ (committed reference outputs)
- examples/run_output_sm/ (superminimal run)

## Deviations
- None. The superminimal parity matches the recorded reference (core identical, edge within tolerance). c3h4 seed-mechanism support added as required.

## Next step
Read prompts/steps/job-07-step-01-statmech.md
