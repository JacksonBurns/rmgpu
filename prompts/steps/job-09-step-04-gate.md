# job-09/step-04: Job-09 gate (adjoint vs finite-difference)

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

- Previous step (already done, its code is in the tree): job-09/step-03-cli
- This step: job-09/step-04-gate
- Next step (do NOT start it): the job gate for job-09 (its step file)

## Goal

Run the job-09 gate: the adjoint vs FD correctness (THE gate -
< 1%), the covariance propagation (the MC cross-check), the performance
(the speedup), the CLI. The "GPU + autodiff" payoff, measured.

## Reference to read (this step's budget)

  (no new reference reads)
  /home/jackson/rmgpu/RMG-Py/examples/rmg/{superminimal,c3h4}/ (the
    small mechanism + the observables)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- gates/gate_09.py: the four checks (the adjoint vs the central
  FD (perturb +/-1e-4, re-run, central diff) - the max rel diff per
  (observable, parameter), target < 1%; the covariance (the +-20% on two
  A's -> the observable covariance vs the MC reference (1000 runs) - the
  one-element diff); the performance (the adjoint pass vs the FD
  equivalent - the speedup); the CLI on an existing run).
- The RMG-Py reference (if applicable - RMG's uncertainty numbers on the
  same mechanism): a comparison (the rmgpu adjoint vs RMG's FD - the
  agreement is the cross-validation; if RMG's is too slow to run, the FD
  reference (this gate's own FD) is the reference - document which).
- reports/job-09.md: the sensitivity matrix + the FD comparison (the
  full matrix, not just the max), the covariance diff, the speedup, the
  torchdae-adjoint status (step 3), the pdep boundary (which params are
  HPL-only).
- A RED (adjoint vs FD > 1%): the adjoint is wrong (or the FD is
  ill-conditioned) - diagnose (the toy ODE from step 1 isolates the
  adjoint plumbing); a RED that persists is a BLOCKER (the adjoint is the
  whole point) - record it precisely.

## Checks (must run and pass before you claim done)

  /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_09.py
    -> the adjoint-vs-FD matrix + the covariance + the speedup recorded
    (target: < 1% adjoint-vs-FD)
  pytest tests/ -q -> all pass

## Pitfalls

- The adjoint-vs-FD is THE gate (the differentiability is the
  payoff) - a > 1% diff is a real bug (the adjoint plumbing or the FD
  conditioning), not a tolerance to relax.
- The FD reference: perturb +/-1e-4, re-run the ODE (a forward solve each
  - the FD is the ground truth here; the adjoint must match it).
- The performance is the "GPU + autodiff" payoff (PLAN.md 1) - even a
  small speedup is recorded (the number is the finding).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-09/step-04: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-09/step-04 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-09-step-04-gate.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
