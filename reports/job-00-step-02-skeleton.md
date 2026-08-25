# job-00/step-02: Package skeleton + CLI stubs + test scaffolding

## What was built

- `pyproject.toml` - project metadata, editable install config, console scripts (rmgpu, rmgpu-version)
- `rmgpu/__init__.py` - exposes `__version__ = "0.1.0"`
- `rmgpu/cli.py` - click group with `run`, `validate` (not implemented), `version` (works) subcommands
- `rmgpu/version.py` - standalone version printer for `rmgpu-version` command
- `rmgpu/{core,molecule,db,ml,kinetics,pdep,statmech,reactor,io,schemas,tools,sensitivity,plugins}/__init__.py` - empty-but-valid subpackages
- `tests/conftest.py` - pytest fixtures: `example_dir`, `ref_db`, `rmgpy`
- `tests/test_smoke.py` - trivial test verifying `rmgpu.__version__ == "0.1.0"`
- `gates/README.md` - gate contract documentation
- `gates/baselines/` - empty directory for reference outputs
- `reports/` - empty directory for gate reports

## Checks run

1. `/home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest tests/ -q`
   - Result: 1 passed in 0.00s

2. `/home/jackson/miniforge3/envs/rmgpu/bin/python -m rmgpu.version`
   - Result: 0.1.0

3. `/home/jackson/miniforge3/envs/rmgpu/bin/rmgpu version`
   - Result: 0.1.0

## Reference reads beyond the list

- Read CLAUDE.md from RMG-Py for project context (discovered automatically)

## Deviations

None.

## What the next step should know first

The package skeleton is bare - no chemistry code. The `rmgpu` conda env has all required dependencies installed. The next step should set up the test infrastructure and create the job-00 gate script.
