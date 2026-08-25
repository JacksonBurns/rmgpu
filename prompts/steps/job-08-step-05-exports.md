# job-08/step-05: Cantera export + Chemkin reader + the export CLI

Job: job-08 - pdep MSC/RS/SLS + isotope + observables + diff/merge + exports
Prereq: job-07 done (CSE works + gated; the network/DoS core)
This step is part of that job. The job's overall goal:
Complete the pressure-dependence method set (MSC, RS, SLS - onto job 07's network/DoS core) and the analysis/interop tooling the parity suite (job 10) needs: isotope support, observables, model diff/merge, and the legacy-format exports (Chemkin read, Cantera YAML, RMS) from the canonical artifact.
The job's gate (run by the job's final step):
gates/gate_08.py: 1. MSC/RS/SLS parity: propane_branching (job 07's setup) with each of the 4 methods in rmgpu vs RMG-Py: per-reaction k(T,P) max relative diff + network-structure match (same tolerances as job 07). 2. Isotope: an isotope example from test/regression (inventory first) in rmgpu vs RMG-Py: species/thermo parity (the ZPE shifts) + mechanism-set parity. 3. Observables: on the c3h4 run (job 06), RMG-Py's reference observables vs rmgpu's: max abs/rel diff per observable type. 4. diff/merge: two job-06 artifacts (superminimal at iter N and N+1): the diff matches RMG-Py's diffmodels on the equivalent models; the merge round-trips (merge(A,B) contains both, no dupes). 5. Export round-trips: core.yaml -> chemkin.inp -> re-read (ckcsvparser) -> core.yaml: species/reaction sets equal, rates match 1e-8. core.yaml -> cantera/chem.yaml: load in Cantera, check species/reaction counts + a sample of NASA coeffs + a sample of Falloff params vs the artifact.

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

- Previous step (already done, its code is in the tree): job-08/step-04-diffmerge
- This step: job-08/step-05-exports
- Next step (do NOT start it): job-08/step-06-gate

## Goal

The interop exports: the Cantera YAML export (ONE route -
mechanism/cantera/chem.yaml, via the Cantera API, not hand-written YAML),
the Chemkin READER (ckcsvparser - round-trips with job 06's writer), the
RMS writer, and the `rmgpu export` CLI.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/tools/canteramodel.py  (888 - the Cantera model
    construction - the EXACT mapping (NASA coeff ranges, the Falloff
    representation (Lindemann vs Troe vs CME)) - port the MAPPING, use
    the Cantera Python API to serialize (do NOT hand-write the YAML))
  RMG-Py/rmgpy/tools/ckcsvparser.py  (251 - the Chemkin READER - port it;
    round-trips with job 06/6's writer)
  RMG-Py's RMS export (the small writer - locate it: grep -l rms
    RMG-Py/rmgpy/tools/)
  (job-06/6's chemkin writer; job-06/5's mechanism schema)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/io/canteramodel.py: `write_cantera(mechanism, path)` ->
  mechanism/cantera/chem.yaml via the Cantera Python API (create the
  mechanism object, serialize) - ONE route (PLAN.md 12.3: drops the
  from_ck/cantera1/cantera2 mess). The mapping (NASA ranges, the Falloff
  representation) per canteramodel.py's logic.
- rmgpu/io/ckcsvparser.py: `read_chemkin(path) -> Mechanism` (the reader -
  round-trips with job 06/6's writer).
- rmgpu/io/rms.py: the RMS writer (the small YAML for
  ReactionMechanismSimulator - port from RMG's).
- rmgpu/cli.py (extend): `rmgpu export <run>/mechanism/core.yaml --to
  chemkin | cantera | rms` (the writers on the artifact).
- tests/test_exports.py: the Cantera export loads in Cantera (the
  species/reaction counts + a NASA sample + a Falloff sample vs the
  artifact); the Chemkin round-trip (core.yaml -> chemkin.inp ->
  read_chemkin -> core.yaml: sets equal, rates 1e-8); the RMS writer
  produces a parseable file.

## Checks (must run and pass before you claim done)

  pytest tests/test_exports.py -q -> all pass
  the Chemkin round-trip: sets equal + rates 1e-8
  the Cantera export loads (counts + NASA + Falloff sample match)

## Pitfalls

- The Cantera mapping is fussy (the NASA coefficient ranges, the
  Falloff representation (Lindemann vs Troe vs CME)) - port canteramodel.
  py's mapping logic, not a hand-written YAML; a Cantera load failure is
  a mapping bug.
- The Chemkin reader + writer (job 06/6) must round-trip EXACTLY (the
  gate step 5 is 1e-8 on rates) - a format drift between them is a bug;
  the round-trip test pins it.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-08/step-05: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-08/step-05 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-08-step-05-exports.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
