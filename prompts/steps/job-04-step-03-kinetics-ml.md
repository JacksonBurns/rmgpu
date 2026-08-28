# job-04/step-03: KineticsML estimator (Chemprop reactions)

Job: job-04 - ML estimators (thermo + kinetics) + rate registry + thesis test
Prereq: jobs 01-02 done (Molecule for structures; rate registry; the ml_estimator block of job 03)
This step is part of that job. The job's overall goal:
The SOLE property estimators, wrapping the two VENDORED checkpoints in this repo's `models/` directory (PLAN.md 3b): the CheMeleon thermo checkpoint (`models/chemeleon_thermo_662946.ckpt`, 9 log10-space targets: log_H298_J_mol, log_S298_J_mol_K, log_Cp_1..7_J_mol_K at 300/400/500/600/800/1000/1500 K) and the Chemprop RIGR reaction checkpoint (`models/chemprop_kinetics_662946.ckpt`, targets log10_A, n, Ea_J_mol; A = per-site pre-exponential in cm^3/(mol*s); input = atom-mapped reaction SMILES). No group additivity, no rate rules, no fallbacks - by design. Where a model does not cover a structure, the system records a COVERAGE GAP (a finding), it does not silently estimate. THE REPLACEMENT: RMG-Py's existing Chemprop-based estimators (rmgpy/ml/estimator.py) are NOT reused - read only for uncertainty-cutoff concepts and the ml_estimator DSL wiring. rmgpu's estimators are re-implemented wrapping the vendored checkpoints with the inference pattern of `models/predict.py` (the model team's own inference code, copied into this repo) - load checkpoint, featurize with the model's featurizer, data.build_dataloader, pl.Trainer(...).predict. LOAD-PATH CONSTRAINT: the checkpoints pickle-reference `models.BoundedOutputTransform` and `models.HuberMetric`, so a module importable as top-level `models` (i.e. `models/models.py`) is required to load them - do not rename/move it. Boundary conversions (raw -> SI, PLAN.md 3b): thermo = 10^pred; kinetics A = 10^pred * degeneracy (CGS), n as-is, Ea as-is. Deliverables: rmgpu/ml/ (base, thermo_estimator, kinetics_estimator), rmgpu/data/estimation.py (the only estimation code), and the thesis-test gate. The rate registry landed in job 02; this job wires ML output into it.
The job's gate (run by the job's final step):
gates/gate_04.py - THE PoC THESIS TEST (PLAN.md 1, 10-Phase 1). Rigorous and honest: 1. Coverage: over the union of species in examples {superminimal, c3h4, minimal, minimal_ml, ethane-oxidation} + all species in primaryThermoLibrary: how many does the thermo model cover? (% + list of uncovered with reasons). Same for reactions (reactant->product pairs from the c3h4/superminimal mechanism-relevant set + the depository subset). 2. Accuracy vs RMG-Py: for every COVERED species (cap ~500, stratified): RMG-Py (library+GA, or ML per minimal_ml config where applicable) gives Hf298/S298/Cp(300,600,1000); compare to rmgpu ML values. Report: N, mean abs err, median, max, p95 for Hf298 (kJ/mol), S298 (J/mol/K), Cp (J/mol/K) + the distributions (histograms in the report). Same for HPL k(T): log10(k_rmgpu/k_rmg) stats at 300/600/1000 K for covered reactions. 3. No-fallback proof: the resolver NEVER falls back (instrumented: count library hits vs ML hits vs coverage errors; the report shows the split). 4. Checkpoint round-trip: BOTH vendored checkpoints (models/chemeleon_thermo_662946.ckpt, models/chemprop_kinetics_662946.ckpt) load via rmgpu.ml.base and reproduce the reference predictions recorded in the step-01 inventory (real checkpoints are present in the tree); tests run the real checkpoints directly (no synthetic test models - dropped 2026-08-28). A RED gate (errors above what makes a mechanism sensible, or coverage below what the test sets need) is a VALID PoC OUTCOME - reported as the thesis- test result with full numbers, recorded in STATUS, user decides. Do not paper over it.

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

- Previous step (already done, its code is in the tree): job-04/step-02-thermo-ml
- This step: job-04/step-03-kinetics-ml
- Next step (do NOT start it): job-04/step-04-registry

## Goal

Build rmgpu's kinetics ML estimator wrapping the vendored Chemprop REACTION
checkpoint (models/chemprop_kinetics_662946.ckpt) - the RIGR reaction model
(featurizer: CondensedGraphOfReactionFeaturizer with RIGR atom/bond
featurizers; see models/models.py) predicting the high-pressure-limit
Arrhenius parameters. This is NEW in the design - RMG's MLEstimator was
species-only; the reaction generalization is exactly what chemprop natively
supports (PLAN.md 6) and the vendored checkpoint proves it: input =
atom-mapped reaction SMILES, outputs = log10_A, n, Ea_J_mol (not a k(T)
grid).

## Reference to read (this step's budget)

  models/predict.py + models/models.py + models/config.py (the vendored
    kinetics checkpoint's featurizer: RIGR_RXN_FEATURIZER =
    CondensedGraphOfReactionFeaturizer(RIGRAtomFeaturizer, RIGRBondFeaturizer);
    the ReactionDatapoint.from_smi(smi, keep_h=True, add_h=True) call; the
    3 target names: log10_A, n, Ea_J_mol)
  models/chemprop_kinetics_662946.ckpt (the real checkpoint; load it)
  chemprop/data/datapoints.py: ReactionDatapoint (the input format;
    atom-mapped SMILES rct>>pdt)
  (the reference predictions from step 1 for the two kinetics reactions)
  RMG-Py/rmgpy/ml/estimator.py  (182 - the species-only OLD estimator: the
    reaction path has NO counterpart in RMG - this is net new; read only for
    the uncertainty-cutoff concepts)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/ml/kinetics_estimator.py:
    `KineticsML(models_dir)` - loads the vendored Chemprop REACTION model
    (models/chemprop_kinetics_662946.ckpt) via base.load_chemprop_model,
    using the model's own RIGR featurizer (RIGR_RXN_FEATURIZER from
    models/models.py - CondensedGraphOfReactionFeaturizer; the reaction
    "mode" is FIXED by the checkpoint, it is not a constructor argument -
    the old plan's RxnMode parameter is gone). The real checkpoint ALWAYS
    exists in the tree (vendored); there is no synthetic reaction model -
    the tests run this real checkpoint directly.
    `predict(reaction, degeneracy) -> KineticsPrediction` where reaction =
    atom-mapped reaction SMILES (reactants>>products; the recipe engine,
    job 05, will produce the atom correspondence - for this step a canonical
    mapping suffices, document what the featurizer expects). Output is
    FIXED by the checkpoint: Arrhenius parameters, NOT a k(T) grid.
    Boundary conversion (PLAN.md 3b): the raw log10_A is the log10 of the
    PER-SITE pre-exponential (library A / degeneracy, in CGS cm^3/(mol*s)) -
    so A_reaction = 10^pred_A * degeneracy; n = pred_n; Ea = pred_Ea (J/mol,
    linear, can be <= 0). Wrap in the registry's Arrhenius expectation
    (job 04's estimation.py consumes).
    Coverage: reactant AND product both covered (reuse the thermo policy
    per-molecule).
- tests/test_kinetics_ml.py: KineticsML on the real vendored checkpoint
  (models/chemprop_kinetics_662946.ckpt) reproduces the step-1 reference
  predictions for the two reactions from predict.py's __main__, with the
  A*degeneracy boundary conversion verified by hand; coverage cases (one
  side uncovered -> MLCoverageError).

## Checks (must run and pass before you claim done)

  pytest tests/test_kinetics_ml.py -q -> all pass
  the vendored checkpoint (models/chemprop_kinetics_662946.ckpt) loads +
    reproduces the step-1 reference predictions via KineticsML

## Pitfalls

- Atom mapping: the reaction featurizer (RIGR, CondensedGraphOfReaction)
  needs atom-mapped reactant>>product SMILES (which atom maps to which) -
  check the featurizer's expected input and document it; RMG's recipe engine
  (job 05) produces the atom correspondence, but for this step a simple
  canonical mapping suffices for the tests.
- The output format is FIXED by the checkpoint (PLAN.md 3b): log10 of the
  PER-SITE A (cm^3/(mol*s), CGS), n, linear Ea (J/mol) - the class converts
  to the registry's expectation (A*degeneracy in CGS; the registry, jobs
  06/07, consumes). Do not invent an A-unit option: the model was trained
  on CGS A (SI m^3 targets were converted to CGS before training).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-04/step-03: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-04/step-03 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-04-step-03-kinetics-ml.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
