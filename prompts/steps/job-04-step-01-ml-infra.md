# job-04/step-01: ML infra: checkpoint inventory + synthetic test model

Job: job-04 - ML estimators (thermo + kinetics) + rate registry + thesis test
Prereq: jobs 01-02 done (Molecule for structures; rate registry; the ml_estimator block of job 03)
This step is part of that job. The job's overall goal:
The SOLE property estimators, built fresh: CheMeleon/Chemprop for thermo (Hf298, S298, Cp(T)) and Chemprop reaction models for high-pressure-limit kinetics. No group additivity, no rate rules, no fallbacks - by design. Where a model does not cover a structure, the system records a COVERAGE GAP (a finding), it does not silently estimate. THE REPLACEMENT: RMG-Py's existing Chemprop-based estimators (rmgpy/ml/estimator.py, a disabled chemprop species wrapper) are NOT reused - they are read only for their checkpoint layout, uncertainty cutoffs, and the ml_estimator DSL wiring. rmgpu's estimators are re-implemented per /home/jackson/rmgpu/chemprop_example/predicting.ipynb (MPNN.load_from_checkpoint + featurizer + MoleculeDataset + build_dataloader + pl.Trainer.predict). Same for checkpoints: the existing checkpoints are consumed via the new estimators' own load path, which mirrors chemprop_example's - not via RMG's MLEstimator. Deliverables: rmgpu/ml/ (base, thermo_estimator, kinetics_estimator), rmgpu/data/estimation.py (the only estimation code), and the thesis-test gate. The rate registry landed in job 02; this job wires ML output into it.
The job's gate (run by the job's final step):
gates/gate_04.py - THE PoC THESIS TEST (PLAN.md 1, 10-Phase 1). Rigorous and honest: 1. Coverage: over the union of species in examples {superminimal, c3h4, minimal, minimal_ml, ethane-oxidation} + all species in primaryThermoLibrary: how many does the thermo model cover? (% + list of uncovered with reasons). Same for reactions (reactant->product pairs from the c3h4/superminimal mechanism-relevant set + the depository subset). 2. Accuracy vs RMG-Py: for every COVERED species (cap ~500, stratified): RMG-Py (library+GA, or ML per minimal_ml config where applicable) gives Hf298/S298/Cp(300,600,1000); compare to rmgpu ML values. Report: N, mean abs err, median, max, p95 for Hf298 (kJ/mol), S298 (J/mol/K), Cp (J/mol/K) + the distributions (histograms in the report). Same for HPL k(T): log10(k_rmgpu/k_rmg) stats at 300/600/1000 K for covered reactions. 3. No-fallback proof: the resolver NEVER falls back (instrumented: count library hits vs ML hits vs coverage errors; the report shows the split). 4. Synthetic-model plumbing: if the real kinetics checkpoint is absent, the synthetic-chemprop reaction model (step 1) round-trips through predict() with exact values (clearly labeled: plumbing, not accuracy). A RED gate (errors above what makes a mechanism sensible, or coverage below what the test sets need) is a VALID PoC OUTCOME - reported as the thesis- test result with full numbers, recorded in STATUS, user decides. Do not paper over it.

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
- This step: job-04/step-01-ml-infra
- Next step (do NOT start it): job-04/step-02-thermo-ml

## Goal

The ML infrastructure shared by both estimators: inspect the existing
checkpoints, pin the inference pattern from chemprop_example, and train a
small SYNTHETIC chemprop model (so the code paths are unit-testable before
the real checkpoints arrive).
This is where the replacement is made concrete: the NEW estimators' load
path is built on chemprop_example's pattern, and RMG-Py's ml/estimator.py is
consulted only for what its checkpoints contain.

## Reference to read (this step's budget)

  /home/jackson/rmgpu/chemprop_example/predicting.ipynb  (190 - THE
    reference: models.MPNN.load_from_checkpoint(checkpoint_path);
    featurizers.SimpleMoleculeMolGraphFeaturizer();
    data.MoleculeDatapoint.from_smi(smi); data.MoleculeDataset(dps,
    featurizer=...); data.build_dataloader(dset, shuffle=False);
    pl.Trainer(logger=None, accelerator=..., devices=1).predict(mpnn,
    loader); np.concatenate(preds))
  /home/jackson/rmgpu/chemprop_example/example_model_v2_regression_mol.ckpt
    (the example checkpoint: load it, inspect what the model expects -
    featurizer type, output dims)
  RMG-Py/rmgpy/ml/estimator.py  (182 - the OLD estimator: read for its
    checkpoint layout (hf298_path, s298_cp_path), uncertainty cutoffs, and
    how the ml_estimator DSL wires it. REPLACED in this job - nothing is
    imported or wrapped from it.)
  RMG-Py/examples/rmg/minimal_ml/input.py  (the ml_estimator DSL wiring)
  reports/job-00.md (the checkpoint locations step 00 located)
  chemprop reaction featurizers (installed; read the API):
    chemprop/featurizers/molgraph/reaction.py - the RxnMode enum
    (REAC_PROD, REAC_PROD_BALANCE, REAC_DIFF, REAC_DIFF_BALANCE, PROD_DIFF,
    PROD_DIFF_BALANCE) + the reaction featurizer classes (for step 3)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- scripts/train_synthetic_chemprop.py: trains TWO small synthetic
  chemprop v2 models (fixed seed, tiny mol set ~50 smiles from the example
  csv or a hardcoded set, random-ish targets) and writes checkpoints to
  tests/fixtures/synthetic/:
    - a SPECIES model (regression, 1 output: a stand-in "property") - for
      thermo plumbing tests
    - a REACTION model (regression, 1 output, reaction-mode featurizer) -
      for kinetics plumbing tests
  The script must be runnable in the env and idempotent (skips if the
  checkpoints exist).
- rmgpu/ml/base.py: `load_chemprop_model(checkpoint_path)` - the single load
  path, per chemprop_example (MPNN.load_from_checkpoint + the model's
  registered featurizer if present else the one passed in) + a
  `predict_raw(model, datapoints)` helper (build_dataloader +
  pl.Trainer(...).predict + concatenate) with device handling (cuda if
  available else cpu; single GPU task rule).
- reports/job-04-step-01.md checkpoint inventory: for each real checkpoint
  located in job 00 (CheMeleon Hf298, S298+Cp; any reaction models on the
  box): path, format, loadable? (try the chemprop_example pattern),
  featurizer type, output dims, and the training UNITS the model predicts in
  (kJ/mol? J/mol/K? kcal? - inspect the checkpoint metadata; if not
  determinable, say so - step 2 converts at the boundary).
- tests/test_ml_base.py: load the synthetic species + reaction models;
  predict on 5 molecules / 2 reactions; shapes + determinism (two runs ->
  same values) + the example checkpoint loads (if the featurizer matches;
  else record why it does not).

## Checks (must run and pass before you claim done)

  /home/jackson/miniforge3/envs/rmgpu/bin/python
    scripts/train_synthetic_chemprop.py -> 2 checkpoints written (or found)
  pytest tests/test_ml_base.py -q -> all pass
  the example checkpoint (example_model_v2_regression_mol.ckpt) loads via
    rmgpu.ml.base (or the report says why not)

## Pitfalls

- The synthetic models are PLUMBING fixtures, not accuracy evidence
  - the report must say so.
- Device: the box shares one GPU (with llama-server) - keep batch sizes
  small, no data loading that thrashes GPU memory; the single-GPU-task rule
  applies to every predict call.
- If a real checkpoint does NOT load via the chemprop_example pattern (e.g.
  a different featurizer than the model expects), record the exact failure
  + what the model's registered featurizer is - that is the finding job 04
  reports, and the new estimator must handle it (the load path takes an
  explicit featurizer argument).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-04/step-01: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-04/step-01 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-04-step-01-ml-infra.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
