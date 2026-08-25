# Gate Contract

Every job gate is a script `gates/gate_NN.py` that:

1. Runs under the `rmgpu` conda environment's Python interpreter
2. Prints `PASS` or `FAIL` to stdout
3. Exits with code 0 (PASS) or 1 (FAIL)
4. Writes a report to `reports/job-NN.md`

The `gates/baselines/` directory stores reference outputs captured from
RMG-Py runs on the same machine, used to verify parity.
