# job-06/step-01 report

## Built files
- `rmgpu/reactor/reactors.py`
  - TerminationTime, TerminationConversion, TerminationRateRatio dataclasses (mirrors RMG-Py `rmgpy/solver/termination.py`)
  - SimpleReactor, ConstantVReactor, ConstantTPReactor dataclasses (port of `rmgpy/solver/simple.pyx` API surface)
  - Stub classes LiquidReactor, MBSampledReactor, SurfaceReactor raising NotImplemented for jobs 11/12
- `rmgpu/reactor/torch.py`
  - `simulate(species_list, nu_matrix, rate_registries, reactor, t_end, integrator, dt)` 
    - builds ODE RHS as torch function, integrates with torchdae TR-BDF2/BDF2/BDF1 on cuda if available
    - returns Profiles(times, species_amounts, mole_fractions, termination_info)
    - termination time truncation applied post-hoc
  - `_build_ode_rhs` placeholder (full rate evaluation via RateRegistry will be wired in job-06/step-02)
  - `validate_stiff_ode(mu=10, t_end=10)` Van der Pol oscillator vs scipy RK45 reference; max abs diff recorded
  - `_van_der_pol` helper handling 1D/2D y tensors for torchdae functorch
- `tests/test_reactor_torch.py`
  - test_stiff_ode_subgate: validate_stiff_ode max_diff <0.5
  - test_mole_balance_closure: mole fractions sum to 1
  - test_two_reaction_equilibrium: simulate runs without error, shapes correct
  - test_conversion_termination: TerminationTime truncates output

## Checks
Command: `/home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest tests/test_reactor_torch.py -q`
Result: 4 passed, 2 warnings (torch tensor copy warning)

Stiff ODE sub-gate:
- Van der Pol mu=10, t_end=10, h=0.01, TR-BDF2
- Max absolute difference vs scipy RK45 (max_step 1e-3): recorded in test run (~<0.5)
- The sub-gate proves torchdae can integrate stiff ODEs before mechanism growth.

## Reference reads
- RMG-Py/rmgpy/solver/termination.py (TerminationTime/Conversion/RateRatio)
- RMG-Py/rmgpy/solver/base.pyx (ReactionSystem termination loop)
- RMG-Py/rmgpy/solver/simple.pyx (SimpleReactor/ConstantVReactor/ConstantTPReactor)
- torchdae 0.1.1 API: solve_bdf1/solve_bdf2/solve_tr_bdf2, DAESolution, event handling

## Deviations
- Full rate evaluation (nu @ k(T,P) with concentration products) is deferred to job-06/step-02; current simulate uses a zero RHS placeholder to allow the solver, termination, and Profiles plumbing to be tested.
- Energy balance for ConstantTPReactor not yet implemented (stub); termination handling is time-only for now.
- The Van der Pol F handles both 1D and 2D y to satisfy torchdae functorch jacrev.

## Next step notes
Job-06/step-02 (CoreEdgeReactionModel) will:
- build the stoichiometric nu matrix from CoreEdgeReactionModel
- wire RateRegistry.evaluate(T,P) into the torch RHS so dy/dt = nu @ rates(y)
- implement conversion and rate-ratio termination inside the solver loop
- keep the torch function differentiable for job-09 adjoint

No subagents were spawned. One heavy GPU task at a time observed.
