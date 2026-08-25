# job-09/step-03: The sensitivity CLI + the report

Job: job-09 - Sensitivity / uncertainty via torchdae adjoint
Prereq: job-06 (reactor), job-04 (the rate registry is differentiable), job-07 (if done - the pdep boundary is documented)
This step is part of that job. The job's overall goal:
Parameter sensitivity + uncertainty quantification via torch autodiff / the torchdae adjoint, replacing RMG's finite-difference + Morris/Sobol tooling. The reactor is already a torch function (job 06), so d(output)/d(parameter) is available - this job exposes it.
The job's gate (run by the job's final step):
gates/gate_09.py: 1. Correctness (THE gate): a small mechanism (superminimal or c3h4) + a few observables + a few rate parameters: the sensitivity via (a) the torchdae ADJOINT vs (b) central finite differences (perturb +/-1e-4, re-run, central diff). Compare: max relative diff per (observable, parameter). TARGET: < 1%. Report the matrix + the FD comparison. 2. Covariance: propagate a small parameter covariance (+-20% on two Arrhenius A's) to the observable covariance (first-order, from the sensitivity matrix - RMG's formula); cross-check ONE element against a Monte-Carlo reference (1000 forward runs, empirical covariance). Report the diff. 3. Performance: the adjoint pass vs the FD equivalent (same observables x parameters) - the speedup (the "GPU + autodiff" payoff - even if small, record it). 4. `rmgpu sensitivity` on an existing run works + writes the files.

## Context (invariant every step)

- You are a FRESH human-started session; you have no other context. This file plus
  what it points at is everything you need.
- Work in /home/jackson/rmgpu/rmgpu (git, branch main). Reference repos (read-
  only, at /home/jackson/rmgpu): RMG-Py, RMG-database, rmgdb, chemprop_example.
- Interpreter: always the env's python directly,
  /home/jackson/miniforge3/envs/rmgpu/bin/python (created in job 00).
  NEVER install into system python or the active venv.
- ONE session at a time; this session MUST NOT spawn subagents. Single heavy
  GPU task at a time on this machine; never kill processes you did not start.
- No Cython, no numba, no QM, no Arkane, no fallback estimators, no second
  reactor backend. SI units internally (J, K, Pa, mol) via pint.
- The existing Chemprop-based estimators in RMG-Py (rmgpy/ml/estimator.py)
  are being REPLACED by rmgpu's new ones - they are reference-only (layout,
  cutoffs, DSL wiring). rmgpu's estimators are re-implemented per
  /home/jackson/rmgpu/chemprop_example/predicting.ipynb. See README.md,
  "The ML estimators are NEW".

## Where you are

- Previous step (already done, its code is in the tree): job-09/step-02-uncertainty
- This step: job-09/step-03-cli
- Next step (do NOT start it): job-09/step-04-gate

## Goal

The `rmgpu sensitivity` CLI (re-run the pass on an existing run -
the mechanism + profiles are in the artifact, no re-run of the mechanism
generation) + the torchdae-adjoint status (the real API, any fallback
decided + documented).

## Reference to read (this step's budget)

  (the sensitivity/uncertainty from steps 1-2; the mechanism
    artifact loader (job 06/5))
  torchdae (the adjoint API - the final status: available/working/buggy)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/cli.py (extend): `rmgpu sensitivity <runDir>` - load the
  run's mechanism + profiles (the artifact), run the sensitivity/
  uncertainty pass (the YAML's uncertainty: block from run.yaml), write
  the uncertainty/ outputs into the run dir (no re-run of the mechanism).
- The torchdae-adjoint status: a definitive statement (the adjoint API
  works / is buggy in X way / is absent) + the decision (use the adjoint
  / the documented fallback) - this is the finding PLAN.md 13 risk 3
  wants resolved.
- tests/test_sensitivity_cli.py: `rmgpu sensitivity` on an existing run
  (a small run from the tests) writes the uncertainty/ files (they
  parse, the matrix is the expected shape).

## Checks (must run and pass before you claim done)

  pytest tests/test_sensitivity_cli.py -q -> all pass
  `rmgpu sensitivity <a small run dir>` -> the uncertainty/ files written

## Pitfalls

- The CLI re-runs on an EXISTING run (the artifact) - the
  mechanism generation is NOT re-run (that is the point - the profiles +
  mechanism are already there); a re-run is a bug.
- The torchdae-adjoint status is a FINDING (PLAN.md 13 risk 3) - a buggy/
  absent adjoint is recorded + the fallback decided, not hidden.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-09/step-03: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-09/step-03 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-09-step-03-cli.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
