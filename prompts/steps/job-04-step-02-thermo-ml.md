# job-04/step-02: ThermoML estimator (CheMeleon) - replacement of RMG's

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

- Previous step (already done, its code is in the tree): job-04/step-01-ml-infra
- This step: job-04/step-02-thermo-ml
- Next step (do NOT start it): job-04/step-03-kinetics-ml

## Goal

Build rmgpu's thermo ML estimator - the REPLACEMENT for RMG's
disabled MLEstimator - wrapping the vendored CheMeleon checkpoint
(models/chemeleon_thermo_662946.ckpt). One checkpoint, 9 log10-space
outputs (log_H298_J_mol, log_S298_J_mol_K, log_Cp_1..7_J_mol_K) - NOT
RMG's two-checkpoint Hf298/S298+Cp layout (see the context block and
PLAN.md 3b). Same uncertainty-cutoff concepts as RMG's MLEstimator, but a
fresh implementation on the models/predict.py inference pattern.

## Reference to read (this step's budget)

  (the checkpoint inventory + reference predictions from step 1 + base.py
    from step 1)
  models/predict.py + models/config.py (the 9 target names; the exact
    predict call the model team uses)
  RMG-Py/rmgpy/ml/estimator.py  (182 - re-read ONLY: the uncertainty
    cutoff concepts, the predict signature semantics. The implementation
    is NEW; its two-checkpoint layout does NOT apply - we have ONE 9-output
    checkpoint.)
  rmgpu/data/thermo.py  (job 02 - the Wilhoit Cp(T) representation the
    estimator must convert into)
  rmgpu/molecule/ (job 01 - Molecule -> SMILES for the featurizer)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/ml/thermo_estimator.py:
    `ThermoML(models_dir)` - loads the vendored CheMeleon checkpoint
    (models/chemeleon_thermo_662946.ckpt) via base.load_chemprop_model.
    The checkpoint ALWAYS exists in the tree (vendored, step 1 verified
    it); there is no synthetic test model - the tests run this real
    checkpoint directly.
    `predict(molecule) -> ThermoPrediction(Hf298, S298, Cp_model,
    uncertainties)` in SI. Boundary conversion (PLAN.md 3b): the raw model
    outputs are log10-space - Hf298 = 10^pred (J/mol), S298 = 10^pred
    (J/mol/K), Cp_i = 10^pred (J/mol/K) at the 7 grid points
    T = 300/400/500/600/800/1000/1500 K. NOTE (document in the report): the
    H298 target was trained on the all-POSITIVE H298 values of the rmgdb
    thermo library, so the model can only predict positive H298; species
    whose true H298 is negative (formation-scale) are outside the model's
    training range - surface this as a coverage/accuracy finding in the
    gate, do not silently use the value.
    Cp model: the checkpoint's 7 discrete Cp values wrapped in a `CpModel`
    that converts to the Wilhoit-compatible Cp(T) representation in
    rmgpu.data.thermo so the rest of the code sees one thermo model
    (fit/interpolate the 7 points into Wilhoit - document the choice +
    the max Cp interpolation error on the grid in the report).
    `.covers(molecule) -> bool` - the coverage policy (heuristic: element
    set / heavy-atom count / SMILES parse; document the exact policy in the
    report). If not covered: raise `MLCoverageError` (captured by callers,
    recorded as a finding; NEVER a silent fallback).
    Batch mode: predict on a list (batching; device per base.py; single
    GPU task rule).
- tests/test_thermo_ml.py: ThermoML on the real vendored checkpoint
  (models/chemeleon_thermo_662946.ckpt): predict on 10 molecules -> values
  through the boundary, SI conversion checked by hand (the 10^pred
  conversion verified against the step-1 recorded raw values for
  CC/CCC/C[CH]CC); covers() policy cases (covered / uncovered element set /
  unparseable); MLCoverageError raised + not swallowed.
- rmgpu/ml/__init__.py: export ThermoML, KineticsML (step 3),
  MLCoverageError, ThermoPrediction, KineticsPrediction.

## Checks (must run and pass before you claim done)

  pytest tests/test_thermo_ml.py -q -> all pass
  ThermoML loads models/chemeleon_thermo_662946.ckpt and reproduces the
    step-1 reference predictions (the checkpoint is vendored in the tree)

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
