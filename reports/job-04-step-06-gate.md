# Job-04 Step-06: Job-04 gate (the thesis test)

GATE STATUS: **RED** (exit 2) - `/home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_04.py`

A RED thesis test is a VALID PoC outcome (step file, Pitfalls): the error
distribution IS the deliverable. The gate was run to completion on the real
checkpoints; the numbers below are the PoC finding.

## What was built

### Files created
- `gates/gate_04.py` - The job-04 gate, implementing the job definition
  (584 lines). Four checks: (1) coverage of the thermo model over the
  example-species union + all primaryThermoLibrary species, and of the
  kinetics model over the c3h4 mechanism-relevant set + depository subset +
  superminimal mechanism-relevant family set (from the committed baselines);
  (2) accuracy vs the RMG-Py reference baselines (Hf298/S298/Cp(300/600/1000)
  per covered species, stratified by class; HPL k(T) at 300/600/1000 K,
  per source bucket) with p95 thresholds; (3) no-fallback proof via
  `EstimationCounts` instrumentation (asserts coverage_errors==0 and that
  library+ML == total); (4) checkpoint round-trip: both vendored checkpoints
  load via `rmgpu.ml.base` and reproduce the step-01 reference predictions
  (`gates/baselines/job04/reference_predictions.json`) at rtol=1e-6,
  atol=1e-4. Writes `reports/gate_04_results.json`; exit 0=PASS, 2=RED.
  Unit conventions handled here: the rmgpu ML kinetics A (CGS cm^3/(mol*s),
  PLAN 3b) is converted to SI before comparing to RMG-Py's `get_rate_coefficient`
  output; dHf298 is reported both signed (J/mol) and absolute.
- `reports/gate_04_results.json` - Full gate output: all four checks,
  per-class and per-bucket accuracy, histograms, the no-fallback split, and
  the checkpoint round-trip detail (this file is the machine-readable record).
- `scripts/thesis_decompose.py` - Analysis tool that decomposes the RED into
  per-parameter errors (fits the reference (A,n,Ea) from the 3 SI k(T) points
  per reaction and compares each to the model's; characterizes dHf298 as
  domain vs offset vs genuine). Cited evidence, kept for reproducibility.
- `reports/thesis_decomposition.json` - Its output: per-parameter kinetics
  deltas (N=287) + per-species thermo deltas (46).
- `reports/job-04.md` - The job-level thesis-test report (checkpoint
  verification, coverage policy, full numbers, no-fallback split, findings).
- `reports/job-04-step-06-gate.md` - This file.

### Files modified
- None (the gate is self-contained; `scripts/reference_props.py` and
  `gates/baselines/thesis_test/` were already committed in the previous
  session as `bde0605`).

## Checks

1. **Gate run (real checkpoints, GPU):**
   `/home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_04.py` -> exit 2,
   **RED**, elapsed 744.3 s. Per-check:
   - `checkpoint_roundtrip` ok - thermo max_abs_diff 0.0, kinetics max_abs_diff
     9.5e-07 (raw log10-space, both real checkpoints, 3 molecules + 2 reactions
     from the step-01 baseline).
   - `thermo_coverage` ok - 46/46 (100%), model covers all, 0 uncovered.
   - `kinetics_coverage` ok - 287/287 (100%).
   - `no_fallback_thermo` ok - split {library: 44, ml: 2, coverage_errors: 0}.
   - `no_fallback_kinetics` ok - split {library: 0, ml: 287, coverage_errors: 0}.
   - `accuracy_hf298` XX - p95 abs 517.0 kJ/mol (threshold 20).
   - `accuracy_s298` XX - p95 abs 33.3 J/(mol*K) (threshold 10).
   - `accuracy_cp` ok - p95 abs 17.1 J/(mol*K) (threshold 20).
   - `accuracy_kT_300` XX - p95 |log10(k ratio)| 16.20 (threshold 1.0).
   - `accuracy_kT_600` XX - p95 8.65.
   - `accuracy_kT_1000` XX - p95 5.43.

2. **Full suite:**
   `/home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest tests/ -q`
   -> `1 failed, 519 passed` in 14.92 s. The single failure,
   `tests/test_ml_base.py::test_single_item_prediction_not_dropped`, is a
   PRE-EXISTING flake, not caused by this step: verified by
   `git stash -u && pytest tests/test_ml_base.py::test_single_item_prediction_
   not_dropped -q && git stash pop` - it fails identically on the clean tree
   (1 failed in 3.61 s), in the same process as the other kinetics-checkpoint
   tests (the documented ~1.6e-2 same-process Ea non-determinism from
   job-04/step-05, references/checkpoint-determinism.md; which test fails
   varies per run). Isolated runs of test_ml_base.py pass 10/10.

## Reference reads beyond the step's list

- `RMG-Py/examples/rmg/{superminimal,c3h4,minimal,minimal_ml,ethane-
  oxidation}/input.py` (listed): verified the example-species union =
  {H2, O2, CH2, C2H2, N2, ethane, Ar} (7 species, all in primaryThermoLibrary
  -> the thermo set is 46 = library + examples).
- `gates/baselines/thesis_test/{thermo,reactions}.json` (committed bde0605):
  46 species, 287 reactions (GRI-Mech3.0-N 256, BurkeH2O2inArHe 17,
  superminimal mechanism-relevant 14).
- `gates/baselines/job04/reference_predictions.json` (step-01): the round-trip
  target.
- `reports/thesis_decomposition.json` (this step's analysis): per-parameter
  decomposition below.

## Deviations

- The gate's accuracy thresholds are PoC sanity floors (p95 Hf298 <= 20 kJ/mol,
  S298 <= 10, Cp <= 20, |log10 k| <= 1), not PLAN numbers - PLAN.md does not
  specify them; the step says a RED is a valid finding. The RED is therefore
  a finding, recorded in STATUS decisions log; the user decides next steps.
- Stratification note: the thermo set is 46 species (cap ~500 not reached -
  the union is simply small), stratified into intermediate/radical/stable
  (12/17/17). The kinetics set is 287, bucketed by source
  (GRI-Mech3.0-N / BurkeH2O2inArHe / superminimal family).
- Three throwaway probes (`scripts/_gate04_smoke.py`, `scripts/_probe_units.py`,
  `scripts/_thermo_char.py`) were committed with the `_` prefix (marked
  scratch in their docstrings): they document the gate's evidence trail
  (resolver-wiring smoke, the CGS/SI settlement of `get_rate_coefficient`,
  per-species Hf298 characterization) and leaving them untracked would keep
  the tree dirty against the done protocol.

## What the next step should know first

**The RED decomposes into two independent, non-wiring model-accuracy findings:**

1. **Kinetics: Ea bias, not A.** The per-parameter fit (287 reactions) shows
   d_log10_A mean -0.04 / p95 5.7, d_n mean +0.03 / p95 1.6 - A and n are
   essentially right - while d_Ea mean **-31.4 kJ/mol (median -29.1, p95 abs
   100.6, max 239.6)**. The log10(k ratio) stats are this single bias read
   through 1/RT: mean 5.49/2.77/1.68 at 300/600/1000 K ~= dEa/(RT ln10).
   The kinetics checkpoint underestimates activation energies systematically.
2. **Thermo: positive-only H298 domain, confirmed.** dHf298 is dominated by
   species with ref Hf298 < 0 (N=12, mean abs 314.8 kJ/mol, all signed
   positive: He +591.1, N2 +324.9, Ar +517.0, HF +448.3, O(S) +475.2, CH4
   +579.3) - exactly the PLAN 3b flagged gap (training on 1662 species all
   with positive H298). Even restricting to the in-domain subset (ref Hf298
   >= 0, N=31) the error is large (mean abs 85.9 kJ/mol, p95 349.0, worst
   cyclopropynylidyne +409.7, C3H2 -149.1), so the model has genuine
   in-domain Hf298 error beyond the domain gap. S298 (p95 33.3) and Cp
   (p95 17.1) are the relatively decent properties.
3. Coverage is 100%/100% and the no-fallback proof holds (0 coverage errors,
   0 third-branch resolutions; thermo split 44 library / 2 ML). The checkpoint
   round-trip is exact (max diff 9.5e-07). **Nothing is broken in the
   rmgpu plumbing; the checkpoints as vendored do not match legacy accuracy.**
   Next steps (user's call): a job-04 fix/decision step, or proceed to job-05
   with the RED recorded as the PoC finding, or a checkpoint retrain request
   to the model team. See reports/job-04.md for the full numbers.
