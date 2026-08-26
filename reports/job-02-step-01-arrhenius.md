# Job-02 Step 01: Arrhenius family and rate-model registry

## Built

- `rmgpu/kinetics/models.py`: Pure-Python rate-expression registry.
  - `KineticsModel`: Common temperature/pressure-validity helpers.
  - `PDepKineticsModel`: Pressure-dependent base class.
  - `Arrhenius`: Modified Arrhenius rate expression with `T0` adjustment.
  - `ArrheniusEP`: Evans-Polanyi activation-energy clamping and conversion to fixed Arrhenius.
  - `make_rate_model(kind, **params)`: Registry constructor.
  - `generate_reverse_rate_coefficient`: Thermodynamic reverse-rate helper.
- `tests/test_kinetics_models.py`: Unit tests for Arrhenius reference agreement, T0 behavior, EP clamping, registry construction, and reverse-rate consistency.

## Checks

- `pytest tests/test_kinetics_models.py -v`: 8 passed.
- RMG-Py reference values generated from three Arrhenius cases using the installed `rmg_env`; comparisons pass with a relative tolerance of `1e-4`.
- Git commit: `a200d55`.

## Deviations

- The step file references a `Tmin/Tmax` extrapolation section for ArrheniusEP, but the RMG-Py reference class does not apply Tmin/Tmax extrapolation in `get_rate_coefficient`. The port therefore preserves the valid-range metadata and rate formula without inventing an extrapolation policy.
- Pyright initially flagged optional dataclass inheritance and method overrides; the implementation retains plain floats and compatible signatures.

## Notes for next step

- Store model parameters as SI floats unless another step deliberately introduces units.
- Preserve the flat registry API for falloff, Chebyshev, Marcus, and tunneling models.
- Use `make_rate_model` plus `evaluate` as the stable interface.
