# job-00/step-01-env Report

## What was built

### scripts/check_env.py
One-off env-verification script that imports every required package, prints their versions, checks `torch.cuda.is_available()`, verifies `rmgdb` is importable, and confirms all five SQLite databases and the checkpoint path exist. Later steps re-run this.

## Checks run

### 1. `/home/jackson/miniforge3/envs/rmgpu/bin/python scripts/check_env.py`

```
numpy: 2.4.6
scipy: 1.17.1
rdkit: 2026.3.5
pint: 0.25.3
chemprop: 2.3.1
cantera: 3.2.0
chemicals: 1.5.2
fluids: 1.3.1
thermo: 0.6.1
torchdae: 0.1.1
sqlalchemy: 2.0.52
polars: 1.44.0
pydantic: 2.13.4
click: 8.4.2
pytest: 9.1.1
torch.cuda.is_available(): True
rmgdb: importable
DB present: /home/jackson/rmgpu/rmgdb/db/thermo.db
DB present: /home/jackson/rmgpu/rmgdb/db/kinetics.db
DB present: /home/jackson/rmgpu/rmgdb/db/transport.db
DB present: /home/jackson/rmgpu/rmgdb/db/solvation.db
DB present: /home/jackson/rmgpu/rmgdb/db/statmech.db
checkpoint present: /home/jackson/rmgpu/chemprop_example/example_model_v2_regression_mol.ckpt
```
All checks pass.

### 2. rmgdb importable

`import rmgdb` succeeds (confirmed in check_env.py output).

### 3. CheMeleon checkpoint located and loadable

- **Path:** `/home/jackson/rmgpu/chemprop_example/example_model_v2_regression_mol.ckpt`
- **Format:** `.ckpt` (PyTorch zip archive)
- **Loaded via chemprop example pattern:** `models.MPNN.load_from_checkpoint(checkpoint_path)` succeeded.
- **Featurizer:** `SimpleMoleculeMolGraphFeaturizer` (from chemprop example).
- **Output dims:** single-task regression model, one prediction per molecule.
- **1-molecule predict test:** ethane (CC) predicted value = `2.166739`.

The checkpoint inventory is complete for job 04 to consume.

## Reference reads beyond the list

None.

## Deviations

None.

## What the next step should know first

The `rmgpu` conda env is fully set up and verified. The `rmgdb` SQLite databases are present and importable. The checkpoint pattern is confirmed working. Next step: build the package skeleton (step 02).
