## job-03/step-03

Built the rmgpu CLI with `run`, `validate`, `schema`, `version`, and stub `import`/`export`/`diff`/`inspect` commands in `rmgpu/cli.py`. Added `examples/minimal.yaml` and `tests/test_cli.py` with 7 tests covering the command behavior. The `validate` command reports every error found in an input file and exits with code 1, while `run` loads, validates, and prints the resolved document. Added a `QuantityDumper` for YAML output and a helper to collect quantity-string errors. Tests pass.
