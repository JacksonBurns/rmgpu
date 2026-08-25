# job-08/step-01: pdep MSC + RS + SLS (onto job-07's core)

Job: job-08 - pdep MSC/RS/SLS + isotope + observables + diff/merge + exports
Prereq: job-07 done (CSE works + gated; the network/DoS core)
This step is part of that job. The job's overall goal:
Complete the pressure-dependence method set (MSC, RS, SLS - onto job 07's network/DoS core) and the analysis/interop tooling the parity suite (job 10) needs: isotope support, observables, model diff/merge, and the legacy-format exports (Chemkin read, Cantera YAML, RMS) from the canonical artifact.
The job's gate (run by the job's final step):
gates/gate_08.py: 1. MSC/RS/SLS parity: propane_branching (job 07's setup) with each of the 4 methods in rmgpu vs RMG-Py: per-reaction k(T,P) max relative diff + network-structure match (same tolerances as job 07). 2. Isotope: an isotope example from test/regression (inventory first) in rmgpu vs RMG-Py: species/thermo parity (the ZPE shifts) + mechanism-set parity. 3. Observables: on the c3h4 run (job 06), RMG-Py's reference observables vs rmgpu's: max abs/rel diff per observable type. 4. diff/merge: two job-06 artifacts (superminimal at iter N and N+1): the diff matches RMG-Py's diffmodels on the equivalent models; the merge round-trips (merge(A,B) contains both, no dupes). 5. Export round-trips: core.yaml -> chemkin.inp -> re-read (ckcsvparser) -> core.yaml: species/reaction sets equal, rates match 1e-8. core.yaml -> cantera/chem.yaml: load in Cantera, check species/reaction counts + a sample of NASA coeffs + a sample of Falloff params vs the artifact.

First step of this job: skim /home/jackson/rmgpu/rmgpu/ORIENTATION.md once
(what stays/goes/external; the master-equation data path; conventions). Later
steps of this job do not need it - everything they need is in their own file.

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

- Previous step (already done, its code is in the tree): (this is the first step of the job)
- This step: job-08/step-01-methods
- Next step (do NOT start it): job-08/step-02-isotopes

## Goal

Port the other three pdep lumping methods (MSC, RS, SLS) onto job
07's network/DoS core - only the collision lumping differs; do NOT
duplicate the network code. The driver's NotImplemented raises (job 07)
become real dispatches.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/pdep/msc.pyx  (186 - MSC: the modified strong-
    collision lumping)
  RMG-Py/rmgpy/pdep/rs.pyx  (232 - RS: the reservoir-state method)
  RMG-Py/rmgpy/pdep/sls.py  (459 - SLS: the simulation-least-squares (the
    ode + matrix-exponential variants - the method strings from job 07))
  (job-07's network.py + collision.py - the core they plug into; the
    driver's dispatch)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/pdep/msc.py, rs.py, sls.py: the three methods, each
  implementing job 07's collision/lumping protocol (the network's rate-
  matrix off-diagonal + the eigenvalue lumping per method).
  The driver (job 07/6) dispatches on the method string (the long strings
  from job 07's table) - the NotImplemented raises for MSC/RS/SLS become
  these.
- The 4 methods share the network/DoS core (job 07) - the test is: each
  method on the SAME network (job 07's toy Lindemann + a propane-
  branching network) produces the method's k(T,P) vs RMG-Py (the gate's
  full parity is step 5's; this step: the toy + 1 real network per
  method).
- tests/test_pdep_methods.py: per method (MSC, RS, SLS-ode, SLS-mexp): the
  toy Lindemann k(T,P) vs RMG-Py (max rel diff recorded) + the
  network-structure invariants hold (the methods don't alter the
  network, only the lumping).

## Checks (must run and pass before you claim done)

  pytest tests/test_pdep_methods.py -q -> all pass
  the toy Lindemann: all 4 methods (CSE + MSC + RS + SLS) vs RMG-Py (max
    rel diff per method recorded)

## Pitfalls

- The methods differ only in the collision lumping (the
  network/DoS/core are job 07's) - do NOT fork the network per method; a
  method that needs a "different network" is a port bug.
- SLS has variants (ode / matrix-exponential - the method strings) - port
  both (the driver dispatches on the exact string).
- The driver must select the method from the YAML (job 03's normalized
  string) - a method the driver can't dispatch is a loud error, not a
  silent CSE.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-08/step-01: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-08/step-01 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-08-step-01-methods.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
