# job-00: Environment, package skeleton, test scaffolding

You are starting implementation of rmgpu (RMG-GPU). Read ORIENTATION.md and PLAN.md
(section 8 for the target layout) before doing anything. This job sets up the
foundation every later job builds on: the conda env, the package skeleton, test
tooling, and the gate harness skeleton. No chemistry yet.

## Context you need

- Workspace: /home/jackson/rmgpu (contains RMG-Py, RMG-database, rmgdb).
- This repo: /home/jackson/rmgpu/rmgpu (git, branch main). You are working HERE.
- rmgdb source: /home/jackson/rmgpu/rmgdb -- read its README and the schema files
  (standard/rmgdb/*/schema.py) now so the package skeleton has the right stubs.
- GPU: this box has CUDA. torch must be the CUDA build. Only ONE heavy GPU task at
  a time on this machine; never kill llama-server or any process you did not start.

## Deliverables

1. **Conda env `rmgpu`** (python 3.11):
   - core: torch (CUDA), numpy, scipy, rdkit, pint, chemprop, cantera,
     chemicals, fluids, thermo, torchdae, sqlalchemy, polars, pydantic, click,
     pytest, pytest-cov
   - Checkpoint availability: verify the CheMeleon checkpoint path used by
     RMG-Py's ml_estimator DSL works (see /home/jackson/rmgpu/RMG-Py/examples/rmg/minimal_ml
     and RMG-Py/rmgpy/ml/estimator.py). Record the exact checkpoint locations you
     found in reports/job-00.md. You do NOT need to make them work yet (that is job 04),
     but the env must be able to import chemprop and load a checkpoint.
   - rmgdb: install from /home/jackson/rmgpu/rmgdb (editable or as a git dependency;
     follow its README). Record the install method in the report.
   - NEVER install into system python or the active venv -- everything must go in the `rmgpu` conda environment.

2. **Package skeleton** at rmgpu/ exactly per ORIENTATION.md layout:
   - pyproject.toml (name rmgpu, version 0.1.0, entry point `rmgpu = rmgpu.cli:main`,
     all deps from the env list)
   - empty-but-valid __init__.py per package; a rmgpu/__init__.py exposing __version__
   - rmgpu/cli.py with a click group `rmgpu` and subcommands `run` (placeholder that
     prints "not yet implemented"), `version`, `validate` (placeholder). Later jobs
     fill these in.
   - rmgpu/schemas/ with pydantic model stubs (job 03 fills them).
   - tests/ with a tests/conftest.py that adds repo root to sys.path and defines
     fixtures: `example_dir` (points at /home/jackson/rmgpu/RMG-Py/examples/rmg),
     `ref_db` (points at /home/jackson/rmgpu/RMG-database), `rmgpy` (path to
     /home/jackson/rmgpu/RMG-Py/rmgpy).
   - gates/ and reports/ directories (gates/ with a README explaining the gate
     contract: every gate script is `gates/gate_NN.py`, runs under conda env rmgpu,
     prints a PASS/FAIL summary, writes reports/job-NN.md; the exit code must be 0
     on PASS and 1 on FAIL so it is CI-able).

3. **Smoke test** (the gate for this job):
   - tests/test_smoke.py: import rmgpu, check version, import every rmgpu subpackage,
     run `rmgpu version` via subprocess, verify `import rdkit, torch, chemprop,
     cantera, chemicals, fluids, thermo, pint, sqlalchemy, torchdae` all work, and
     (CUDA box) verify `torch.cuda.is_available()` is True.
   - Run the full test suite. All green.

## Gate (job 00)

  /home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest tests/ -q     -> all pass
  /home/jackson/miniforge3/envs/rmgpu/bin/python version                  -> prints 0.1.0
Write reports/job-00.md: env versions (pip freeze | grep -E 'torch|rdkit|chemprop|...'),
checkpoint locations found, rmgdb install method, and the pytest summary line.

## When done

- Update STATUS.md: flip job 00 row to done, append session log entry with commit
  hashes and "next: start job-01".
- Commit with message "job-00: env, skeleton, test scaffolding".
- STOP. Do not start job 01.

## Notes / pitfalls

- torchdae on PyPI is 0.1.1 (verified in the plan). If the install pulls something
  else, record the version; the API surface we need is BDF1/BDF2/TR-BDF2/Radau-IIA +
  index reduction + adjoint (PLAN.md 5).
- If `torch` resolves to CPU-only wheels, fix it (cu12x/cu13 index) and re-check
  `torch.cuda.is_available()`.
- Keep the skeleton BARE. Do not pre-write chemistry code; later prompts specify
  what goes in each module.
