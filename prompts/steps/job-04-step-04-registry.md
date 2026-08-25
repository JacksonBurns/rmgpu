# job-04/step-04: Rate registry: tunneling + forward/reverse wiring

Job: job-04 - ML estimators (thermo + kinetics) + rate registry + thesis test
Prereq: jobs 01-02 done (Molecule for structures; rate registry; the ml_estimator block of job 03)
This step is part of that job. The job's overall goal:
The SOLE property estimators, built fresh: CheMeleon/Chemprop for thermo (Hf298, S298, Cp(T)) and Chemprop reaction models for high-pressure-limit kinetics. No group additivity, no rate rules, no fallbacks - by design. Where a model does not cover a structure, the system records a COVERAGE GAP (a finding), it does not silently estimate. THE REPLACEMENT: RMG-Py's existing Chemprop-based estimators (rmgpy/ml/estimator.py, a disabled chemprop species wrapper) are NOT reused - they are read only for their checkpoint layout, uncertainty cutoffs, and the ml_estimator DSL wiring. rmgpu's estimators are re-implemented per /home/jackson/rmgpu/chemprop_example/predicting.ipynb (MPNN.load_from_checkpoint + featurizer + MoleculeDataset + build_dataloader + pl.Trainer.predict). Same for checkpoints: the existing checkpoints are consumed via the new estimators' own load path, which mirrors chemprop_example's - not via RMG's MLEstimator. Deliverables: rmgpu/ml/ (base, thermo_estimator, kinetics_estimator), rmgpu/data/estimation.py (the only estimation code), and the thesis-test gate. The rate registry landed in job 02; this job wires ML output into it.
The job's gate (run by the job's final step):
gates/gate_04.py - THE PoC THESIS TEST (PLAN.md 1, 10-Phase 1). Rigorous and honest: 1. Coverage: over the union of species in examples {superminimal, c3h4, minimal, minimal_ml, ethane-oxidation} + all species in primaryThermoLibrary: how many does the thermo model cover? (% + list of uncovered with reasons). Same for reactions (reactant->product pairs from the c3h4/superminimal mechanism-relevant set + the depository subset). 2. Accuracy vs RMG-Py: for every COVERED species (cap ~500, stratified): RMG-Py (library+GA, or ML per minimal_ml config where applicable) gives Hf298/S298/Cp(300,600,1000); compare to rmgpu ML values. Report: N, mean abs err, median, max, p95 for Hf298 (kJ/mol), S298 (J/mol/K), Cp (J/mol/K) + the distributions (histograms in the report). Same for HPL k(T): log10(k_rmgpu/k_rmg) stats at 300/600/1000 K for covered reactions. 3. No-fallback proof: the resolver NEVER falls back (instrumented: count library hits vs ML hits vs coverage errors; the report shows the split). 4. Synthetic-model plumbing: if the real kinetics checkpoint is absent, the synthetic-chemprop reaction model (step 1) round-trips through predict() with exact values (clearly labeled: plumbing, not accuracy). A RED gate (errors above what makes a mechanism sensible, or coverage below what the test sets need) is a VALID PoC OUTCOME - reported as the thesis- test result with full numbers, recorded in STATUS, user decides. Do not paper over it.

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

- Previous step (already done, its code is in the tree): job-04/step-03-kinetics-ml
- This step: job-04/step-04-registry
- Next step (do NOT start it): job-04/step-05-estimation

## Goal

Complete the rate registry (job 02 built the core): tunneling
factors (Wigner at minimum), the forward/reverse thermodynamic-consistency
helper wired to real thermo models (job 02's), and the registry as the
single consumer-facing interface for reactor/ME (jobs 06/07).

## Reference to read (this step's budget)

  RMG-Py/rmgpy/kinetics/tunneling.pyx  (265 - Wigner at minimum;
    Eckart only if RMG-Py exposes it publicly)
  RMG-Py/rmgpy/kinetics/model.pyx  (578 - generate_reverse_rate_
    coefficient: dG(T) integration for thermodynamic consistency; job 02
    ported the math - this step wires it to job 02's real thermo models)
  rmgpu/kinetics/models.py + rmgpu/data/thermo.py (job 02)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/kinetics/models.py (extend):
    Tunneling: Wigner factor (Ea-dependent) applied to k(T); Eckart if
    exposed (decision recorded).
    forward/reverse: the registry's reverse-rate generation wired to
    rmgpu.data.thermo (dG(T) from the actual Wilhoit/NASA models) -
    replace job 02's duck-typed test stand-in.
    `RateModel.evaluate(T, P)`, `.forward()`, `.reverse(thermo_reactants,
    thermo_products)` - the surface jobs 06/07/09 consume.
- tests: Wigner factor values vs RMG-Py (script); reverse-rate consistency:
  for a toy reaction with hand-computed thermo, k_reverse from the registry
  == RMG-Py's generate_reverse_rate_coefficient (same params, tol 1e-10);
  the full registry: every model type evaluates k(T,P) (a table in the
  test).
- A registry "contract" docstring at the top of models.py: what the reactor
  (job 06) and the ME (job 07) call, with the SI conventions.

## Checks (must run and pass before you claim done)

  pytest tests/test_kinetics_registry.py -q -> all pass
  reverse-rate: registry vs RMG-Py on the toy reaction (diff recorded)

## Pitfalls

- The registry is consumed by THREE jobs (06 reactor, 07 ME, 09
  sensitivity) - the API must be flat, explicit, and documented now; a
  churn here is a churn in three places.
- Do NOT add a second evaluation path - evaluate(T,P) is the one way in.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-04/step-04: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-04/step-04 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-04-step-04-registry.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
