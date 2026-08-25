# job-04/step-02: ThermoML estimator (CheMeleon) - replacement of RMG's

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

- Previous step (already done, its code is in the tree): job-04/step-01-ml-infra
- This step: job-04/step-02-thermo-ml
- Next step (do NOT start it): job-04/step-03-kinetics-ml

## Goal

Build rmgpu's thermo ML estimator - the REPLACEMENT for RMG's
disabled MLEstimator. Same checkpoint layout (Hf298 model + S298+Cp model),
same uncertainty-cutoff concepts, but a fresh implementation on the
chemprop_example inference pattern.

## Reference to read (this step's budget)

  (the checkpoint inventory from step 1 + base.py from step 1)
  RMG-Py/rmgpy/ml/estimator.py  (182 - re-read ONLY: the two-checkpoint
    layout (hf298_path, s298_cp_path), the uncertainty cutoffs, the
    predict signature semantics. The implementation is NEW.)
  rmgpu/data/thermo.py  (job 02 - the Wilhoit Cp(T) representation the
    estimator must convert into)
  rmgpu/molecule/ (job 01 - Molecule -> SMILES for the featurizer)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/ml/thermo_estimator.py:
    `ThermoML(models_dir)` - loads the CheMeleon checkpoints (Hf298 model;
    S298+Cp model) via base.load_chemprop_model (the chemprop_example
    pattern), matching RMG's layout. If the real checkpoints are not yet
    loadable (step 1's inventory said why not), the class is written to
    their SPEC (documented) and unit-tested against the synthetic species
    model (mapped to the two-output layout).
    `predict(molecule) -> ThermoPrediction(Hf298, S298, Cp_model,
    uncertainties)` in SI (convert the checkpoint's training units at the
    boundary - step 1's inventory pins the units per checkpoint).
    Cp model: whatever the checkpoint predicts (T-grid or coefficients -
    documented per checkpoint) wrapped in a `CpModel` that converts to the
    Wilhoit-compatible Cp(T) representation in rmgpu.data.thermo so the
    rest of the code sees one thermo model.
    `.covers(molecule) -> bool` - the coverage policy (heuristic: element
    set / heavy-atom count / SMILES parse; document the exact policy in the
    report). If not covered: raise `MLCoverageError` (captured by callers,
    recorded as a finding; NEVER a silent fallback).
    Batch mode: predict on a list (batching; device per base.py; single
    GPU task rule).
- tests/test_thermo_ml.py: synthetic-model plumbing (predict on 10 molecules
  -> exact values through the boundary, SI conversion checked by hand),
  covers() policy cases (covered / uncovered element set / unparseable),
  MLCoverageError raised + not swallowed.
- rmgpu/ml/__init__.py: export ThermoML, KineticsML (step 3),
  MLCoverageError, ThermoPrediction, KineticsPrediction.

## Checks (must run and pass before you claim done)

  pytest tests/test_thermo_ml.py -q -> all pass
  ThermoML loads the real checkpoints if step 1 said loadable (else the
    synthetic path + the BLOCKED-until-checkpoint note)

## Pitfalls

- Checkpoint-agnostic: a new checkpoint (better model) must be a
  config change (the ml_estimator: block), not a code change (PLAN.md 8a.3
  - the checkpoint interface is the seam).
- The uncertainty cutoffs from RMG's MLEstimator: port the CONCEPT (a
  prediction above the cutoff is flagged) - the values are per-checkpoint
  metadata if present, else documented defaults.
- Do NOT import rmgpy or RMG's ml.estimator anywhere - the replacement is
  the point.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-04/step-02: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-04/step-02 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-04-step-02-thermo-ml.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
