# job-09: Sensitivity / uncertainty via torchdae adjoint

Read ORIENTATION.md and PLAN.md section 5 (sensitivity -> torch autodiff/adjoint),
13 (risk: torchdae maturity). Prereqs: job-06 (reactor works), job-04 (rate registry
differentiable).

## Goal

Parameter sensitivity and uncertainty quantification using torch autodiff / the
torchdae adjoint, replacing RMG's finite-difference + Morris/Sobol tooling. The
reactor is already a torch function, so d(output)/d(parameter) is available for
free - this job exposes it.

## Reference (read)

  RMG-Py/rmgpy/tools/globaluncertainty.py (28k) - RMG's uncertainty framework
    (Morris screening, Sobol, covariance propagation, the observable-to-parameter
    pipeline). Port the SEMANTICS (which parameters, which observables, how
    results are reported), not the finite-difference numerics.
  RMG-Py/rmgpy/tools/uncertainty.py (if present) / rmgpy/data/... uncertainty
    handling: parameter distributions (Arrhenius A/n/Ea uncertainties, the
    `uncertainties` stored on rate models - job 04 kept these as storage types).
  The `uncertainty:` YAML block (job 03 schema): which species/parameters, enabled.

## Deliverables

1. `rmgpu/sensitivity/` package:
   - `sensitivity.py`: for a given run (mechanism + reactor + T,P), compute
     d(observable)/d(parameter) for the selected parameters via the torchdae
     ADJOINT (not forward-mode - one adjoint pass per observable, cheap for many
     parameters). Parameters: rate-model params (A, n, Ea per reaction, with
     uncertainties), and thermo params (Hf298, S298 per species) where the model
     supports it. Output: a sensitivity matrix (observable x parameter) + the
     covariance propagated to the observables.
   - `uncertainty.py`: propagate parameter covariance to observable covariance
     (first-order, from the sensitivity matrix - port RMG's formula), write
     `uncertainty/covariance.csv` + `parameter_uncertainties.json` (PLAN.md 12.3).
   - Morris/Sobol screening: port the SCREENING SEMANTICS (which parameters to
     screen, the design), but implement with the adjoint (each Morris trajectory
     is a forward+adjoint pair) - faster than RMG's finite differences.
2. Wire into the run: when the YAML `uncertainty:` block is enabled, after the
   final iteration, run the sensitivity/uncertainty pass and write the
   uncertainty/ outputs.
3. CLI: `rmgpu sensitivity <runDir>` (re-run the pass on an existing run without
   re-running the mechanism generation - the mechanism + profiles are in the
   artifact).

## Gate (job 09) -> gates/gate_09.py, report reports/job-09.md

  1. Correctness (the key gate): pick a small mechanism (superminimal or c3h4,
     pdep on if job 07 done) + a few observables + a few rate parameters. Compute
     the sensitivity via (a) the torchdae ADJOINT and (b) central finite
     differences (perturb param by +/-1e-4, re-run the ODE, central difference).
     Compare: max relative diff per (observable, parameter). TARGET: < 1% (the
     adjoint must be correct; this is the whole point of the GPU/differentiable
     reactor). Report the matrix + the FD comparison.
  2. Covariance: propagate a small parameter covariance (e.g. +-20% on two
     Arrhenius A's) to the observable covariance via the sensitivity matrix;
     cross-check ONE element against a Monte-Carlo reference (1000 forward runs
     with random params, compute the empirical covariance). Report the diff.
  3. Performance: time the adjoint sensitivity pass vs the FD equivalent for the
     same (observables x parameters) - report the speedup (this is the "GPU +
     autodiff" payoff, PLAN.md section 1). Even if small, record it.
  4. `rmgpu sensitivity` on an existing run works and writes the files.

## When done

STATUS.md + commit "job-09: sensitivity/uncertainty via torchdae adjoint" + STOP.

## Pitfalls

- The adjoint requires the ODE right-hand side to be a differentiable torch
  function (it is, from job 06). But the ME / pdep path (job 07) may not be
  differentiable yet - if a pressure-dependent reaction's k(T,P) is not a smooth
  torch function of the parameters, SENSITIVITY is defined for the HPL params only
  and the pdep-fitted params are treated as inputs (document this boundary in the
  report). Do NOT claim full differentiability through the ME unless you verified it.
- torchdae adjoint API: confirm the installed version supports the adjoint/solver
  API you need (PLAN.md 13 flags torchdae 0.1.1 as young). If the adjoint is not
  available or is buggy, fall back to RECURRENT forward-backward (RNN-style) or
  finite differences and DOCUMENT it as a torchdae limitation (this is a finding,
  record it - do not silently ship FD as if it were the adjoint).
- Numerical conditioning: sensitivities of fast reactions can be huge/stiff; use
  the same tolerances as the forward solve and report conditioning warnings.
