# job-04/step-01 - ML infra: vendored-checkpoint verification + load path

Date: 2026-08-28

## What was built

- `scripts/record_reference_predictions.py` - generates the non-circular
  reference baseline: runs BOTH vendored checkpoints through models/predict.py's
  OWN predictor classes (the model team's code, not rmgpu.ml.base) on the fixed
  set (molecules CC, CCC, C[CH]CC; the two reaction SMILES from
  models/predict.py's `__main__`, extracted from that file's AST at run time so
  the inputs can never drift from the model team's demo). Writes
  `gates/baselines/job04/reference_predictions.json` (raw log10-space values,
  full float precision). Idempotent: re-run prints "baseline unchanged" when
  values are identical, or a loud WARNING + rewrite on drift.
- `gates/baselines/job04/reference_predictions.json` - the committed baseline
  (see numbers below).
- `rmgpu/ml/base.py` - the single shared load path:
  - `load_chemprop_model(path)` / `load_thermo_checkpoint()` /
    `load_kinetics_checkpoint()` -> `ChempropCheckpoint` (model in eval mode +
    the checkpoint's featurizer + ordered target names). Unknown checkpoint
    file -> ValueError (no guessing; the checkpoint interface is the only
    model-improvement seam, PLAN.md 8a.3).
  - `ChempropCheckpoint.predict_raw(datapoints, batch_size=8)` - the model
    team's inference pattern: chemprop dataset with the checkpoint's featurizer,
    `data.build_dataloader`, `pl.Trainer(accelerator=..., devices=1,
    logger=False, enable_progress_bar=False).predict`, `torch.cat`, CPU
    float32 tensor, one row per input in input order. Values are RAW
    (model-native log10 space); SI boundary conversion is the estimators' job
    (step 02/03).
  - `get_device()` - cuda if available else cpu (single-GPU-task rule noted).
- `tests/test_ml_base.py` - 10 tests, all on the REAL vendored checkpoints.

## Load-path mechanics (recorded per the step file)

The checkpoints pickle-reference the top-level module `models`
(`models.BoundedOutputTransform`, `models.HuberMetric` in `models/models.py`).
`models/` has no `__init__.py`, so the models/ DIRECTORY ITSELF must be on
`sys.path`: `rmgpu/ml/base.py` inserts `<repo>/models` at `sys.path[0]` before
the first load (`_ensure_models_importable`), which makes `import models`
resolve to `models/models.py` (and `import config` / `import predict` to the
siblings - exactly the mechanism `models/predict.py` itself relies on). A
cached top-level `models` module is accepted only if it actually carries
`BoundedOutputTransform` (a bare namespace package from a shadowing dir is
purged and re-imported). The directory must stay at the repo root named
`models`; `models.py` must not be renamed/moved or those classes altered.

## Checkpoint inventory (verified in the rmgpu env, this session)

Both load and predict via the models/predict.py pattern in
`/home/jackson/miniforge3/envs/rmgpu` (torch 2.13.0+cu130, CUDA available,
chemprop 2.3.1, lightning 2.6.5, python 3.11.16).

### models/chemeleon_thermo_662946.ckpt (~40 MB)
- Loadable: YES (verified this session, CUDA).
- Featurizer: `SimpleMoleculeMolGraphFeaturizer` (explicit H).
- Outputs: 9, all log10-space, names in models/config.py:
  `log_H298_J_mol, log_S298_J_mol_K, log_Cp_1..7_J_mol_K`.
- Unit contract (PLAN.md 3b): value = 10^pred -> H298 J/mol, S298 J/mol/K,
  Cp 7-point grid at 300/400/500/600/800/1000/1500 K in J/mol/K. Trained on
  1662 rmgdb thermo-library species (all H298 positive -> H298 target is
  log10 of that positive value; the model can only predict H298 > 0).

### models/chemprop_kinetics_662946.ckpt (~2.6 MB)
- Loadable: YES (verified this session, CUDA).
- Featurizer: `CondensedGraphOfReactionFeaturizer` (RIGR atom + bond
  featurizers); input = atom-mapped reaction SMILES.
- Outputs: 3, models/config.py: `log10_A, n, Ea_J_mol`.
- Unit contract (PLAN.md 3b): `log10_A` = log10 of the PER-SITE pre-exponential
  in CGS cm^3/(mol*s) (A_reaction = 10^pred * degeneracy); `n` as-is; `Ea`
  linear J/mol (kept linear so Ea<=0 chemistry survives). Trained on rmgdb
  kinetics-library HPL parameters.

### Reference predictions (committed baseline; full precision in the JSON)

Thermo (molecules CC, CCC, C[CH]CC), log10-space:

| smiles | log_H298 | log_S298 | log_Cp_1 | log_Cp_2 | log_Cp_3 | log_Cp_4 | log_Cp_5 | log_Cp_6 | log_Cp_7 |
|---|---|---|---|---|---|---|---|---|---|
| CC  | 5.407312 | 2.401812 | 1.723495 | 1.808566 | 1.879084 | 1.930409 | 2.010036 | 2.059148 | 2.134799 |
| CCC | 4.505925 | 2.428415 | 1.855146 | 1.961992 | 2.044192 | 2.103522 | 2.188794 | 2.244889 | 2.316439 |
| C[CH]CC | 4.272232 | 2.511820 | 1.966797 | 2.065605 | 2.141296 | 2.198658 | 2.274433 | 2.327942 | 2.391295 |

Kinetics (the two __main__ reaction SMILES):

| reaction (truncated) | log10_A | n | Ea_J_mol |
|---|---|---|---|
| [O:1]([C:2]([C:3]([C:4](=[O:5])... (butane + O2 -> acetone + acetic acid + formaldehyde) | 17.095583 | 0.380845 | 225706.578125 |
| [O:1]=[c:2]1[n:3]([H:7])[c:4]... (isoxazole + O2 -> HCN + CO2) | 12.490026 | 0.375581 | 83949.187500 |

Sane-scale check (also asserted in the test suite): 10^log_H298(CC) ~ 2.56e5
J/mol (O(100) kJ/mol), S298 ~ 252 J/mol/K, Cp(300K) ~ 52.9 J/mol/K; Ea both
real barriers (225.7 kJ/mol, 83.9 kJ/mol).

## Checks run (real results)

- `/home/jackson/miniforge3/envs/rmgpu/bin/python scripts/record_reference_predictions.py`
  -> first run: wrote `gates/baselines/job04/reference_predictions.json` (both
  checkpoints loaded + predicted on GPU). Second run: `baseline unchanged:
  gates/baselines/job04/reference_predictions.json` (idempotent, deterministic).
- `/home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest tests/test_ml_base.py -q`
  -> **10 passed in 4.17s** (both real checkpoints load via rmgpu.ml.base with
  correct featurizer/target contracts; 3x9 and 2x3 values match the committed
  baseline at tol 1e-4; determinism; unknown-checkpoint rejection; single-item
  no-drop regression; device check).
- Full suite `/home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest tests/ -q`
  -> **482 passed** (was 472 at job-03 close; no regressions).

## Reference reads beyond the list

- `chemprop/data/dataloader.py::build_dataloader` (installed package,
  ~96 lines) - to pin down the `drop_last` auto-behavior (see deviations).
  No other out-of-budget reads.

## Deviations

1. **`drop_last=False` made explicit in `predict_raw`** (models/predict.py
   leaves it auto). chemprop's auto mode drops the last batch when
   `len(dataset) % batch_size == 1` - at inference with batch_size=8 a single
   input would be SILENTLY lost (observed: "Dropping last batch of size 1"
   with a 1-element dataset). Values for multi-item sets are identical
   (verified: baseline generated with the auto path matches base.py's
   explicit path to 1e-4), but dropping a row is a correctness hazard, so
   base.py pins `drop_last=False` + a regression test
   (`test_single_item_prediction_not_dropped`). This is a deliberate,
   documented deviation from the reference pattern, in the reference's favor
   for batch sizes that divide the set evenly.
2. **Determinism tolerance, not bitwise.** The step says "two runs -> same
   values"; on CUDA the MPNN forward is not bit-deterministic. Measured spread
   over two runs of the same inputs: thermo max abs diff 4.77e-7 (rel
   ~1.1e-7), kinetics 0.0. The test asserts `allclose(atol=1e-6, rtol=0)`,
   which is ~50x tighter than the baseline tolerance (1e-4) and matches the
   "same values" intent; on CPU the runs are bit-identical.
3. **Reaction inputs extracted by AST** from models/predict.py's `__main__`
   instead of re-typing the two SMILES literals (a manual transcription
   produced a malformed SMILES on the first attempt - RDKit parse error -
   which is exactly the drift risk the AST extraction eliminates).
4. `scripts/` is not a package; the baseline script does its own
   `sys.path` mechanics and is meant to be run directly (not imported).

## What step 02 (thermo estimator) should know first

- Build on `rmgpu.ml.base.load_thermo_checkpoint()` ->
  `ChempropCheckpoint.predict_raw(molecule_datapoints)`. The datapoints are
  chemprop's `data.MoleculeDatapoint.from_smi(smi, keep_h=True, add_h=True)`
  (explicit H; the featurizer expects them). Raw outputs are log10-space:
  H298 J/mol = 10^col0, S298 J/mol/K = 10^col1, Cp grid = 10^cols2..8 at
  300/400/500/600/800/1000/1500 K.
- The model was trained on rmgdb thermo-library SMILES (rmgpu Molecule
  canonicalization, OpenBabel for N/S per job-01). The H298 output is
  bounded > 0 by construction (log10 of positive values) - species with
  Hf < 0 will be a finding, not an error to "fix".
- Batch sizes: keep small (8 is the base default) - single-GPU-task rule;
  the estimator should predict in chunks and never hold more than a few
  thousand molecules in one dataset.
- The uncertainty-cutoff concepts from RMG's rmgpy/ml/estimator.py (read
  2026-08-28, 182 lines, reference-only): its `MLEstimator` has NO numeric
  cutoff - it wraps two Chemprop checkpoints (Hf298; S298+Cp) with
  uncertainty=0 hardcoded, and the `mlEstimator(thermo=True, minHeavyAtoms=4)`
  DSL block (examples/rmg/minimal_ml/input.py) gates on `minHeavyAtoms` only
  - ML is used for species with >= 4 heavy atoms, library otherwise. That
  `minHeavyAtoms` gate is the only real cutoff concept to carry into the
  estimators; the two-checkpoint layout does NOT apply to our 9-output model.
