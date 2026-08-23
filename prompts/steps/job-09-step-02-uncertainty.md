# job-09/step-02: Covariance propagation + the run wiring + Morris/Sobol

Job: job-09 - Sensitivity / uncertainty via torchdae adjoint
Prereq: job-06 (reactor), job-04 (the rate registry is differentiable), job-07 (if done - the pdep boundary is documented)
This step is part of that job. The job's overall goal:
Parameter sensitivity + uncertainty quantification via torch autodiff / the torchdae adjoint, replacing RMG's finite-difference + Morris/Sobol tooling. The reactor is already a torch function (job 06), so d(output)/d(parameter) is available - this job exposes it.
The job's gate (run by the job's final step):
gates/gate_09.py: 1. Correctness (THE gate): a small mechanism (superminimal or c3h4) + a few observables + a few rate parameters: the sensitivity via (a) the torchdae ADJOINT vs (b) central finite differences (perturb +/-1e-4, re-run, central diff). Compare: max relative diff per (observable, parameter). TARGET: < 1%. Report the matrix + the FD comparison. 2. Covariance: propagate a small parameter covariance (+-20% on two Arrhenius A's) to the observable covariance (first-order, from the sensitivity matrix - RMG's formula); cross-check ONE element against a Monte-Carlo reference (1000 forward runs, empirical covariance). Report the diff. 3. Performance: the adjoint pass vs the FD equivalent (same observables x parameters) - the speedup (the "GPU + autodiff" payoff - even if small, record it). 4. `rmgpu sensitivity` on an existing run works + writes the files.

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

- Previous step (already done, its code is in the tree): job-09/step-01-sensitivity
- This step: job-09/step-02-uncertainty
- Next step (do NOT start it): job-09/step-03-cli

## Goal

The uncertainty side: propagate the parameter covariance to the
observable covariance (first-order, RMG's formula) + write the
uncertainty/ outputs + the Morris/Sobol screening (the semantics, via the
adjoint) + wire it into the run (the YAML uncertainty: block).

## Reference to read (this step's budget)

  RMG-Py/rmgpy/tools/globaluncertainty.py (re-read: the covariance
    propagation formula (first-order, from the sensitivity matrix) + the
    Morris/Sobol screening design (which parameters to screen, the design)
    - port the semantics)
  (job 09/1's sensitivity; job 08/3's observables; job 03's uncertainty:
    block)
  (PLAN.md 12.3: the uncertainty/ output files - covariance.csv +
    parameter_uncertainties.json)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/sensitivity/uncertainty.py:
    `propagate(sensitivity_matrix, parameter_covariance) ->
    observable_covariance` (the first-order propagation - RMG's formula:
    cov_Y = S cov_X S^T where S is the sensitivity matrix);
    the Morris/Sobol screening: the SCREENING semantics (which parameters,
    the design) but via the adjoint (each Morris trajectory is a
    forward+adjoint pair - faster than RMG's finite differences);
    the output writers: uncertainty/covariance.csv +
    uncertainty/parameter_uncertainties.json (PLAN.md 12.3).
- The run wiring (job 06/3's main.py): when the YAML uncertainty: block
  is enabled, after the final iteration, run the sensitivity/uncertainty
  pass + write the uncertainty/ outputs (the mechanism + profiles are
  already in the run).
- tests/test_uncertainty.py: a small parameter covariance on the toy
  system: the propagated covariance vs a Monte-Carlo reference (100
  forward runs, the empirical covariance - the one-element cross-check
  the gate generalizes to 1000); the Morris screening runs (the design +
  the per-parameter effect).

## Checks (must run and pass before you claim done)

  pytest tests/test_uncertainty.py -q -> all pass
  the covariance: the propagated vs the MC reference (the diff recorded)

## Pitfalls

- The first-order propagation is RMG's formula (S cov S^T) - a
  different formula (e.g. a second-order term) changes the result; port
  RMG's, verify against the MC reference (the gate's cross-check).
- The Morris/Sobol is the SEMANTICS (which params, the design) via the
  adjoint - not RMG's finite-difference implementation; the speedup is the
  payoff (record it).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-09/step-02: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-09/step-02 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-09-step-02-uncertainty.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
