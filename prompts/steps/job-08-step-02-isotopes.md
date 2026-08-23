# job-08/step-02: Isotope support

Job: job-08 - pdep MSC/RS/SLS + isotope + observables + diff/merge + exports
Prereq: job-07 done (CSE works + gated; the network/DoS core)
This step is part of that job. The job's overall goal:
Complete the pressure-dependence method set (MSC, RS, SLS - onto job 07's network/DoS core) and the analysis/interop tooling the parity suite (job 10) needs: isotope support, observables, model diff/merge, and the legacy-format exports (Chemkin read, Cantera YAML, RMS) from the canonical artifact.
The job's gate (run by the job's final step):
gates/gate_08.py: 1. MSC/RS/SLS parity: propane_branching (job 07's setup) with each of the 4 methods in rmgpu vs RMG-Py: per-reaction k(T,P) max relative diff + network-structure match (same tolerances as job 07). 2. Isotope: an isotope example from test/regression (inventory first) in rmgpu vs RMG-Py: species/thermo parity (the ZPE shifts) + mechanism-set parity. 3. Observables: on the c3h4 run (job 06), RMG-Py's reference observables vs rmgpu's: max abs/rel diff per observable type. 4. diff/merge: two job-06 artifacts (superminimal at iter N and N+1): the diff matches RMG-Py's diffmodels on the equivalent models; the merge round-trips (merge(A,B) contains both, no dupes). 5. Export round-trips: core.yaml -> chemkin.inp -> re-read (ckcsvparser) -> core.yaml: species/reaction sets equal, rates match 1e-8. core.yaml -> cantera/chem.yaml: load in Cantera, check species/reaction counts + a sample of NASA coeffs + a sample of Falloff params vs the artifact.

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

- Previous step (already done, its code is in the tree): job-08/step-01-methods
- This step: job-08/step-02-isotopes
- Next step (do NOT start it): job-08/step-03-observables

## Goal

Isotope handling: isotope-aware species/thermo - the parity suite
includes isotope examples. Port RMG's mass/isotopic-species logic + the
thermo adjustments (the ZPE shifts per isotope - RMG's empirical/
statistical approach, no QM).

## Reference to read (this step's budget)

  RMG-Py/rmgpy/tools/isotopes.py  (882 - the isotope handling:
    the isotope-aware species, the thermo adjustments (the ZPE shifts per
    isotope - RMG's reduced-mass/symmetry corrections from the statmech
    layer - port the approach, VERIFY against RMG-Py's numbers, do not
    derive your own))
  RMG-Py/rmgpy/molecule/molecule.py (the isotope representation - RMG
    carries isotopes on the atoms; rmgpu's Molecule (job 01) - check it
    carries isotopes; if not, add the minimum)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/tools/isotopes.py:
    the isotope-aware species: the Molecule's isotope data (job 01 - add
    if missing), the isotopic-species construction (the isotope variant of
    a species), the thermo adjustments (the ZPE shifts per isotope - RMG's
    corrections, verified against RMG-Py).
    The isotope spec in the YAML (the species: block's isotope field - if
    job 03's schema lacks it, add the minimum + note it) - the isotope
    examples (test/regression) use it.
- tests/test_isotopes.py: an isotope species (e.g. 13C-ethane or the
  example's isotope): the thermo (the ZPE shift) vs RMG-Py (the species +
  the shifted thermo - max diff recorded); the mechanism: an isotope
  reaction's rate vs RMG-Py (the isotope effect on the rate - if RMG's
  isotope handling affects rates, port that too - check what the example
  exercises).

## Checks (must run and pass before you claim done)

  pytest tests/test_isotopes.py -q -> all pass
  the isotope species: thermo + (if applicable) rate vs RMG-Py (diffs
    recorded)

## Pitfalls

- Isotope thermo is empirical/statistical (NOT QM) - verify
  against RMG-Py's numbers; do not derive your own corrections (the gate
  step 2 compares to RMG-Py).
- The Molecule's isotope representation: if job 01 did not carry isotopes,
  add the minimum (the isotope number per atom) - note it in the report;
  the recipe engine (job 05) must not be broken by the addition.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-08/step-02: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-08/step-02 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-08-step-02-isotopes.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
