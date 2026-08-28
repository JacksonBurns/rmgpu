# job-04/step-02: ThermoML estimator (CheMeleon)

## What was built

- `rmgpu/ml/thermo_estimator.py`: `ThermoML` estimator wrapping the CheMeleon checkpoint. Implements `predict()`, `covers()`, `ThermoPrediction`, `CpModel`, `WilhoitModel`, and `MLCoverageError`. Converts raw log10-space predictions to SI units (J/mol, J/mol/K).
- `tests/test_thermo_ml.py`: Six tests covering checkpoint loading, coverage policy, prediction accuracy (verified against reference predictions with tol 1e-4), and `MLCoverageError` raising.
- `rmgpu/ml/__init__.py`: Exports `ThermoML` and `MLCoverageError` from the thermo estimator.

## Checks run

```
pytest tests/test_thermo_ml.py -q
```

Result: **6 passed in 3.67s**

## Reference reads

- `models/predict.py` (inference pattern: Chemprop load_model, build_dataloader, Trainer.predict)
- `models/config.py` (target names: log_H298_J_mol, log_S298_J_mol_K, log_Cp_1..7_J_mol_K)
- `rmgpy/ml/estimator.py` (uncertainty cutoff concepts; not reused)
- `rmgpu/data/thermo.py` (Wilhoit model for Cp interpolation)
- `rmgpu/molecule/molecule.py` (Molecule.to_smiles())
- `rmgpu/ml/base.py` (load_thermo_checkpoint, ChempropCheckpoint, predict_raw)

## Deviations

- `ALLOWED_ELEMENTS` hardcoded instead of using `Chem.GetPeriodicTable().GetElements()` (RDKit API changed; hardcoded set covers all standard elements).
- `WilhoitModel.get_enthalpy()` and `get_entropy()` return 0.0 (H0, S0 constants not determined; Wilhoit fit simplified to Cp0=Cp[0], CpInf=Cp[-1], a0=a1=a2=a3=0.0).

## What the next step should know

The next step is `job-04/step-03-kinetics-ml`, which will implement the `KineticsML` estimator. It will follow the same pattern as `ThermoML` but wrap the Chemprop kinetics checkpoint. Key context:

- The kinetics checkpoint predicts `log10_A`, `n`, and `Ea_J_mol`.
- Boundary conversion: `A = 10^pred * degeneracy` (CGS), `n` and `Ea` as-is.
- The `KineticsML` class will need to handle reaction SMILES (atom-mapped) instead of molecule SMILES.
- The `MLCoverageError` will be raised for uncovered reactions.
- The `CpModel` and `WilhoitModel` are not needed for kinetics; a different `RateModel` will be required.
