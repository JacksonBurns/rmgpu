# job-09/step-01: The adjoint sensitivity engine

Job: job-09 - Sensitivity / uncertainty via torchdae adjoint
Prereq: job-06 (reactor), job-04 (the rate registry is differentiable), job-07 (if done - the pdep boundary is documented)
This step is part of that job. The job's overall goal:
Parameter sensitivity + uncertainty quantification via torch autodiff / the torchdae adjoint, replacing RMG's finite-difference + Morris/Sobol tooling. The reactor is already a torch function (job 06), so d(output)/d(parameter) is available - this job exposes it.
The job's gate (run by the job's final step):
gates/gate_09.py: 1. Correctness (THE gate): a small mechanism (superminimal or c3h4) + a few observables + a few rate parameters: the sensitivity via (a) the torchdae ADJOINT vs (b) central finite differences (perturb +/-1e-4, re-run, central diff). Compare: max relative diff per (observable, parameter). TARGET: < 1%. Report the matrix + the FD comparison. 2. Covariance: propagate a small parameter covariance (+-20% on two Arrhenius A's) to the observable covariance (first-order, from the sensitivity matrix - RMG's formula); cross-check ONE element against a Monte-Carlo reference (1000 forward runs, empirical covariance). Report the diff. 3. Performance: the adjoint pass vs the FD equivalent (same observables x parameters) - the speedup (the "GPU + autodiff" payoff - even if small, record it). 4. `rmgpu sensitivity` on an existing run works + writes the files.

First step of this job: skim /home/jackson/rmgpu/rmgpu/ORIENTATION.md once
(what stays/goes/external; the master-equation data path; conventions). Later
steps of this job do not need it - everything they need is in their own file.

## Context (invariant every step)

- You are a FRESH subagent session; you have no other context. This file plus
  what it points at is everything you need.
- Work in /home/jackson/rmgpu/rmgpu (git, branch main). Reference repos (read-
  only, at /home/jackson/rmgpu): RMG-Py, RMG-database, rmgdb, chemprop_example.
- Interpreter: always the env's python directly,
  /home/jackson/miniforge3/envs/rmgpu/bin/python (created in job 00).
  NEVER install into system python or the active venv.
- ONE subagent at a time; this session MUST NOT spawn subagents. Single heavy
  GPU task at a time on this machine; never kill processes you did not start.
- No Cython, no numba, no QM, no Arkane, no fallback estimators, no second
  reactor backend. SI units internally (J, K, Pa, mol) via pint.
- The existing Chemprop-based estimators in RMG-Py (rmgpy/ml/estimator.py)
  are being REPLACED by rmgpu's new ones - they are reference-only (layout,
  cutoffs, DSL wiring). rmgpu's estimators are re-implemented per
  /home/jackson/rmgpu/chemprop_example/predicting.ipynb. See README.md,
  "The ML estimators are NEW".

## Where you are

- Previous step (already done, its code is in the tree): (this is the first step of the job)
- This step: job-09/step-01-sensitivity
- Next step (do NOT start it): job-09/step-02-uncertainty

## Goal

The sensitivity core: d(observable)/d(parameter) via the torchdae
ADJOINT (not forward-mode - one adjoint pass per observable, cheap for
many parameters). Parameters: the rate-model params (A, n, Ea per
reaction, with the uncertainties from job 02/04) + the thermo params
(Hf298, S298 per species) where the model supports it.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/tools/globaluncertainty.py  (508 - RMG's
    uncertainty framework: the Morris screening, the Sobol, the covariance
    propagation, the observable-to-parameter pipeline - port the
    SEMANTICS (which parameters, which observables, how results are
    reported), NOT the finite-difference numerics)
  RMG-Py/rmgpy/tools/uncertainty.py (if present - the parameter
    distributions: the Arrhenius A/n/Ea uncertainties stored on the rate
    models - job 02/04 kept them as storage types)
  (job-06's reactor torch function - the differentiable ODE rhs; the
    registry's params)
  torchdae (the adjoint API - confirm the installed version supports the
    adjoint/solver API; PLAN.md 13 flags 0.1.1 as young - if the adjoint
    is not available/buggy: the fallback is documented, step 3's)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/sensitivity/sensitivity.py:
    `sensitivity(mechanism, reactor, T, P, parameters, observables) ->
    SensResult` (the observable x parameter matrix):
      the parameters: the rate-model params (A, n, Ea per reaction - the
      registry's params, with the stored uncertainties) + the thermo
      params (Hf298, S298 per species) where the model supports it;
      the observables: from job 08/3's observables (conversion, time-
      to-x, the integrals) - the d(observable)/d(param) via the torchdae
      ADJOINT (one adjoint pass per observable - cheap for many params);
      the torchdae adjoint: the ODE rhs is a differentiable torch function
      (job 06) - the params are leaf tensors, the adjoint backprops through
      the solve.
    The pdep boundary: if a pressure-dependent reaction's k(T,P) is not a
    smooth torch function of the params (job 07's fit may not be
    differentiable), the sensitivity is defined for the HPL params only +
    the pdep-fitted params are treated as INPUTS (document this boundary
    in the report - do NOT claim differentiability through the ME unless
    verified).
- tests/test_sensitivity.py: a toy ODE (a 2-reaction system with a known
  analytic sensitivity): the adjoint sensitivity vs the analytic value
  (max rel diff < 1e-3 - pins the adjoint plumbing before the gate's FD
  comparison); the parameter plumbing (a param that should not affect an
  observable gives ~0).

## Checks (must run and pass before you claim done)

  pytest tests/test_sensitivity.py -q -> all pass
  the toy ODE: adjoint sensitivity vs the analytic value (< 1e-3)

## Pitfalls

- The adjoint requires the ODE rhs to be differentiable through
  the params (job 06's torch function - confirm the params are leaf
  tensors in the graph; if the registry's numpy path breaks the graph,
  the params must flow through torch - note it).
- torchdae 0.1.1's adjoint is the risk (PLAN.md 13 risk 3) - if it is
  unavailable/buggy, the fallback (recurrent forward-backward or FD) is a
  DOCUMENTED finding (step 3 decides), not a silent substitution.
- The pdep boundary is a REAL limitation (the ME fit may not be
  differentiable) - document it precisely; a false "fully differentiable"
  claim is a finding error.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-09/step-01: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-09/step-01 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-09-step-01-sensitivity.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
