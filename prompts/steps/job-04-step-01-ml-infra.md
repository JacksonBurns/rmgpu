# job-04/step-01: ML infra: vendored-checkpoint verification + synthetic test model

Job: job-04 - ML estimators (thermo + kinetics) + rate registry + thesis test
Prereq: jobs 01-02 done (Molecule for structures; rate registry; the ml_estimator block of job 03)
This step is part of that job. The job's overall goal:
The SOLE property estimators, wrapping the two VENDORED checkpoints in this repo's `models/` directory (PLAN.md 3b): the CheMeleon thermo checkpoint (`models/chemeleon_thermo_122e91.ckpt`, 9 log10-space targets: log_H298_J_mol, log_S298_J_mol_K, log_Cp_1..7_J_mol_K at 300/400/500/600/800/1000/1500 K) and the Chemprop RIGR reaction checkpoint (`models/chemprop_kinetics_122e91.ckpt`, targets log10_A, n, Ea_J_mol; A = per-site pre-exponential in cm^3/(mol*s); input = atom-mapped reaction SMILES). No group additivity, no rate rules, no fallbacks - by design. Where a model does not cover a structure, the system records a COVERAGE GAP (a finding), it does not silently estimate. THE REPLACEMENT: RMG-Py's existing Chemprop-based estimators (rmgpy/ml/estimator.py) are NOT reused - read only for uncertainty-cutoff concepts and the ml_estimator DSL wiring. rmgpu's estimators are re-implemented wrapping the vendored checkpoints with the inference pattern of `models/predict.py` (the model team's own inference code, copied into this repo) - load checkpoint, featurize with the model's featurizer, data.build_dataloader, pl.Trainer(...).predict. LOAD-PATH CONSTRAINT: the checkpoints pickle-reference `models.BoundedOutputTransform` and `models.HuberMetric`, so a module importable as top-level `models` (i.e. `models/models.py`) is required to load them - do not rename/move it. Boundary conversions (raw -> SI, PLAN.md 3b): thermo = 10^pred; kinetics A = 10^pred * degeneracy (CGS), n as-is, Ea as-is. Deliverables: rmgpu/ml/ (base, thermo_estimator, kinetics_estimator), rmgpu/data/estimation.py (the only estimation code), and the thesis-test gate. The rate registry landed in job 02; this job wires ML output into it.
The job's gate (run by the job's final step):
gates/gate_04.py - THE PoC THESIS TEST (PLAN.md 1, 10-Phase 1). Rigorous and honest: 1. Coverage: over the union of species in examples {superminimal, c3h4, minimal, minimal_ml, ethane-oxidation} + all species in primaryThermoLibrary: how many does the thermo model cover? (% + list of uncovered with reasons). Same for reactions (reactant->product pairs from the c3h4/superminimal mechanism-relevant set + the depository subset). 2. Accuracy vs RMG-Py: for every COVERED species (cap ~500, stratified): RMG-Py (library+GA, or ML per minimal_ml config where applicable) gives Hf298/S298/Cp(300,600,1000); compare to rmgpu ML values. Report: N, mean abs err, median, max, p95 for Hf298 (kJ/mol), S298 (J/mol/K), Cp (J/mol/K) + the distributions (histograms in the report). Same for HPL k(T): log10(k_rmgpu/k_rmg) stats at 300/600/1000 K for covered reactions. 3. No-fallback proof: the resolver NEVER falls back (instrumented: count library hits vs ML hits vs coverage errors; the report shows the split). 4. Checkpoint round-trip: BOTH vendored checkpoints (models/chemeleon_thermo_122e91.ckpt, models/chemprop_kinetics_122e91.ckpt) load via rmgpu.ml.base and reproduce the reference predictions recorded in the step-01 inventory (real checkpoints are present in the tree); the synthetic-chemprop models (step 1) still round-trip through predict() with exact values for the plumbing tests (clearly labeled: plumbing, not accuracy). A RED gate (errors above what makes a mechanism sensible, or coverage below what the test sets need) is a VALID PoC OUTCOME - reported as the thesis- test result with full numbers, recorded in STATUS, user decides. Do not paper over it.

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
- The two ML checkpoints are VENDORED in this repo's models/ directory
  (PLAN.md 3b; the models/ dir also holds the model team's inference code,
  models/predict.py, which is the authoritative inference reference).
  RMG-Py's old Chemprop wrapper (rmgpy/ml/estimator.py) is reference-only
  (uncertainty-cutoff concepts, ml_estimator DSL wiring). The checkpoints
  require a module importable as top-level 'models' (models/models.py) to
  load - do not rename/move it. See README.md,
  "The ML estimators are NEW and the checkpoints are REAL".

## Where you are

- Previous step (already done, its code is in the tree): (this is the first step of the job)
- This step: job-04/step-01-ml-infra
- Next step (do NOT start it): job-04/step-02-thermo-ml

## Goal

The ML infrastructure shared by both estimators: document the vendored
checkpoints' contracts, pin the inference pattern from models/predict.py
(the model team's own code, now in this repo), and train a small SYNTHETIC
chemprop model (so the code paths are unit-testable with exact expected
values, independent of the real checkpoints).
The real checkpoints ALREADY EXIST in models/ (vendored, PLAN.md 3b) - the
"checkpoint inventory" of the old plan is now a verification: confirm both
load, record their featurizer/output contracts, and record reference
predictions for a fixed set of molecules/reactions.
This is where the replacement is made concrete: the NEW estimators' load
path follows models/predict.py, and RMG-Py's ml/estimator.py is consulted
only for its uncertainty-cutoff concepts and DSL wiring.

## Reference to read (this step's budget)

  models/predict.py  (100 - THE reference: the model team's predictor classes;
    chemprop.models.utils.load_model(ckpt); MoleculeDatapoint/ReactionDatapoint
    .from_smi(smi, keep_h=True, add_h=True); MoleculeDataset/ReactionDataset
    with the checkpoint's featurizer; data.build_dataloader;
    pl.Trainer(accelerator=..., devices=1, logger=False).predict)
  models/models.py  (84 - the featurizer instances + BoundedOutputTransform +
    HuberMetric; why the module must stay importable as top-level `models`)
  models/config.py  (the target names: THERMO_TARGETS 9, KINETICS_TARGETS 3)
  models/chemeleon_thermo_122e91.ckpt + models/chemprop_kinetics_122e91.ckpt
    (the real checkpoints; load them, verify the featurizers/outputs)
  RMG-Py/rmgpy/ml/estimator.py  (182 - the OLD estimator: read ONLY for its
    uncertainty-cutoff concepts and how the ml_estimator DSL names checkpoints.
    REPLACED in this job - nothing is imported or wrapped from it. Note its
    two-checkpoint Hf298/S298+Cp layout does NOT match the vendored single
    9-output thermo checkpoint.)
  RMG-Py/examples/rmg/minimal_ml/input.py  (the ml_estimator DSL wiring)
  reports/job-00.md (the old checkpoint inventory - the example checkpoint;
    the real ones are now vendored in models/)

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
  path, per models/predict.py (chemprop.models.utils.load_model; the module
  must be importable as top-level `models` first - document/ensure the
  import mechanics, e.g. sys.path insertion of the repo's models/ dir or an
  alias, and record the exact mechanism in the report) + a
  `predict_raw(model, datapoints, featurizer)` helper (dataset +
  build_dataloader + pl.Trainer(...).predict + concatenate) with device
  handling (cuda if available else cpu; single GPU task rule).
- reports/job-04-step-01.md checkpoint inventory: for each vendored
  checkpoint in models/ (chemeleon_thermo_122e91.ckpt,
  chemprop_kinetics_122e91.ckpt): path, loadable (verified in this env),
  featurizer type, output dims + target names (models/config.py), and the
  exact unit contract (PLAN.md 3b: thermo log10 of J/mol, J/mol/K, Cp 7-pt
  grid at 300/400/500/600/800/1000/1500 K; kinetics log10 of per-site A in
  cm^3/(mol*s) + n + linear Ea J/mol). Also record REFERENCE PREDICTIONS:
  run models/predict.py (or base.py) on a FIXED set (e.g. CC, CCC, C[CH]CC,
  and the two reaction SMILES from predict.py's __main__) and commit the
  numbers to reports/ or gates/baselines/ so step 2/3 and the gate can
  assert the estimators reproduce them.
- tests/test_ml_base.py: load the synthetic species + reaction models;
  predict on 5 molecules / 2 reactions; shapes + determinism (two runs ->
  same values). Also: both REAL vendored checkpoints load via rmgpu.ml.base
  and the reference predictions match the recorded values (tol 1e-4; if a
  checkpoint does not load, record the exact failure - but they should, it
  was verified before this job).

## Checks (must run and pass before you claim done)

  /home/jackson/miniforge3/envs/rmgpu/bin/python
    scripts/train_synthetic_chemprop.py -> 2 checkpoints written (or found)
  pytest tests/test_ml_base.py -q -> all pass (includes both real
    checkpoints loading via rmgpu.ml.base)
  the reference-prediction numbers are recorded in the report/baselines

## Pitfalls

- The synthetic models are PLUMBING fixtures, not accuracy evidence
  - the report must say so.
- Device: the box shares one GPU (with llama-server) - keep batch sizes
  small, no data loading that thrashes GPU memory; the single-GPU-task rule
  applies to every predict call.
- If a real checkpoint does NOT load via the models/predict.py pattern in
  this env (it should - verified during plan update), record the exact
  failure + the model's registered featurizer - that is a finding for the
  report; the new load path must still work (it follows models/predict.py).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-04/step-01: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-04/step-01 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-04-step-01-ml-infra.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
