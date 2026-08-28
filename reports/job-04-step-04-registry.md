# job-04/step-04: Rate registry: tunneling + forward/reverse wiring

## Built

### rmgpu/kinetics/models.py (extended)

- Added `generate_reverse_rate_coefficient_with_thermo(k_forward, thermo_reactants, thermo_products, T)` function that computes thermodynamic consistency using actual thermo models. This function takes thermo model objects with `get_enthalpy(T)` and `get_entropy(T)` methods, calculates dH_rxn and dS_rxn from them, and applies the standard thermodynamic consistency formula.

- Added `RateRegistry` dataclass that serves as the single consumer-facing interface for reactor/ME (jobs 06/07/09). It has:
  - `forward_model`: The forward rate model (KineticsModel)
  - `tunneling_model`: Optional tunneling model (Wigner or Eckart)
  - `evaluate(T, P)`: Evaluates forward rate coefficient with tunneling correction
  - `forward()`: Returns the forward rate model
  - `reverse(thermo_reactants, thermo_products, T)`: Generates reverse rate coefficient using actual thermo models

### tests/test_kinetics_registry.py (new)

Comprehensive tests for the rate registry and thermodynamic consistency:

- `test_wigner_factor_matches_rmgpy`: Verifies Wigner tunneling factor matches RMG-Py implementation exactly
- `test_eckart_tunneling_factor`: Tests Eckart tunneling factor calculation
- `test_reverse_rate_coefficient_basic`: Tests basic reverse rate coefficient calculation
- `test_reverse_rate_with_thermo_models`: Tests reverse rate coefficient with thermo models
- `test_registry_evaluate_with_tunneling`: Tests registry evaluation with tunneling correction
- `test_registry_reverse_rate`: Tests registry reverse rate generation
- `test_all_model_types_evaluate`: Tests that all model types can evaluate k(T,P)
- `test_registry_interface`: Tests the registry interface methods

## Checks run

### pytest tests/test_kinetics_registry.py -v

All 8 tests passed:

```
tests/test_kinetics_registry.py::test_wigner_factor_matches_rmgpy PASSED [ 12%]
tests/test_kinetics_registry.py::test_eckart_tunneling_factor PASSED     [ 25%]
tests/test_kinetics_registry.py::test_reverse_rate_coefficient_basic PASSED [ 37%]
tests/test_kinetics_registry.py::test_reverse_rate_with_thermo_models PASSED [ 50%]
tests/test_kinetics_registry.py::test_registry_evaluate_with_tunneling PASSED [ 62%]
tests/test_kinetics_registry.py::test_registry_reverse_rate PASSED [ 75%]
tests/test_kinetics_registry.py::test_all_model_types_evaluate PASSED [ 87%]
tests/test_kinetics_registry.py::test_registry_interface PASSED [100%]
```

## Reference reads

- Read RMG-Py/rmgpy/kinetics/tunneling.pyx (Wigner and Eckart implementations)
- Read RMG-Py/rmgpy/kinetics/model.pyx (KineticsModel and PDepKineticsModel base classes)
- Read rmgpu/kinetics/models.py (existing implementation)
- Read rmgpu/data/thermo.py (Wilhoit and NASA models)
- Searched for generate_reverse_rate_coefficient in RMG-Py (found it was already implemented in rmgpu)

## Deviations

None. The implementation follows the step file requirements exactly.

## What the next step should know first

The next step should know that:

1. The rate registry (`RateRegistry`) is now the single consumer-facing interface for reactor/ME (jobs 06/07/09). It provides `evaluate(T, P)`, `forward()`, and `reverse(thermo_reactants, thermo_products, T)` methods.

2. The reverse rate coefficient is now wired to actual thermo models via `generate_reverse_rate_coefficient_with_thermo()`. This function expects thermo model objects with `get_enthalpy(T)` and `get_entropy(T)` methods.

3. Tunneling corrections (Wigner and Eckart) are applied automatically when a tunneling model is provided to the registry.

4. The registry uses RMG-Py's R constant (8.314472) for thermodynamic consistency calculations.

5. The `RateRegistry` dataclass uses `Optional[object]` for the tunneling model to support both Wigner and Eckart tunneling models (and any future tunneling models).
