# job-09: Sensitivity / uncertainty via torchdae adjoint

Status of this file: a JOB BRIEF, not a task. Do not try to "do this job"
in one session - it is decomposed into the steps below, one session each.
The human reads STATUS.md's NEXT pointer; this file is the map of the job.

## Goal

Parameter sensitivity + uncertainty quantification via torch autodiff / the torchdae adjoint, replacing RMG's finite-difference + Morris/Sobol tooling. The reactor is already a torch function (job 06), so d(output)/d(parameter) is available - this job exposes it.

## Prereq

job-06 (reactor), job-04 (the rate registry is differentiable), job-07 (if done - the pdep boundary is documented)

## Steps (strictly sequential; one fresh human-started session each)

  step 01  prompts/steps/job-09-step-01-sensitivity.md  The adjoint sensitivity engine
  step 02  prompts/steps/job-09-step-02-uncertainty.md  Covariance propagation + the run wiring + Morris/Sobol
  step 03  prompts/steps/job-09-step-03-cli.md  The sensitivity CLI + the report
  step 04  prompts/steps/job-09-step-04-gate.md  Job-09 gate (adjoint vs finite-difference)

## The job gate

Run by the final step's session (gates/gate_09.py, report
reports/job-09.md):

gates/gate_09.py: 1. Correctness (THE gate): a small mechanism (superminimal or c3h4) + a few observables + a few rate parameters: the sensitivity via (a) the torchdae ADJOINT vs (b) central finite differences (perturb +/-1e-4, re-run, central diff). Compare: max relative diff per (observable, parameter). TARGET: < 1%. Report the matrix + the FD comparison. 2. Covariance: propagate a small parameter covariance (+-20% on two Arrhenius A's) to the observable covariance (first-order, from the sensitivity matrix - RMG's formula); cross-check ONE element against a Monte-Carlo reference (1000 forward runs, empirical covariance). Report the diff. 3. Performance: the adjoint pass vs the FD equivalent (same observables x parameters) - the speedup (the "GPU + autodiff" payoff - even if small, record it). 4. `rmgpu sensitivity` on an existing run works + writes the files.

## When the job is done

The final step's report (reports/job-09.md) has the gate result, the
job table row is `done` (or `blocked` with the cause), and the NEXT pointer
in STATUS.md targets job-10's first step (if there is a next job).
