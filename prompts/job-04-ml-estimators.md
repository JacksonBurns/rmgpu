# job-04: ML estimators (thermo + kinetics) + rate registry + thesis test

Status of this file: a JOB BRIEF, not a task. Do not try to "do this job"
in one session - it is decomposed into the steps below, one session each.
The human reads STATUS.md's NEXT pointer; this file is the map of the job.

## Goal

The SOLE property estimators, wrapping the two VENDORED checkpoints in this repo's
`models/` directory (PLAN.md 3b): the CheMeleon thermo checkpoint
(`models/chemeleon_thermo_122e91.ckpt`, 9 log10-space targets: log_H298_J_mol,
log_S298_J_mol_K, log_Cp_1..7_J_mol_K at 300/400/500/600/800/1000/1500 K) and the
Chemprop RIGR reaction checkpoint (`models/chemprop_kinetics_122e91.ckpt`, targets
log10_A, n, Ea_J_mol; A = per-site pre-exponential in cm^3/(mol*s); input = atom-mapped
reaction SMILES). No group additivity, no rate rules, no fallbacks - by design. Where a
model does not cover a structure, the system records a COVERAGE GAP (a finding), it does
not silently estimate. THE REPLACEMENT: RMG-Py's existing Chemprop-based estimators
(rmgpy/ml/estimator.py) are NOT reused - read only for uncertainty-cutoff concepts and
the ml_estimator DSL wiring. rmgpu's estimators are re-implemented wrapping the vendored
checkpoints with the inference pattern of `models/predict.py` (the model team's own
inference code, copied into this repo) - load checkpoint, featurize with the model's
featurizer, data.build_dataloader, pl.Trainer(...).predict. LOAD-PATH CONSTRAINT: the
checkpoints pickle-reference `models.BoundedOutputTransform` and `models.HuberMetric`,
so a module importable as top-level `models` (i.e. `models/models.py`) is required to
load them - do not rename/move it. Boundary conversions (raw -> SI, PLAN.md 3b): thermo
= 10^pred; kinetics A = 10^pred * degeneracy (CGS), n as-is, Ea as-is. Deliverables:
rmgpu/ml/ (base, thermo_estimator, kinetics_estimator), rmgpu/data/estimation.py (the
only estimation code), and the thesis-test gate. The rate registry landed in job 02; this
job wires ML output into it.

## Prereq

jobs 01-02 done (Molecule for structures; rate registry; the ml_estimator block of job 03)

## Steps (strictly sequential; one fresh human-started session each)

  step 01  prompts/steps/job-04-step-01-ml-infra.md  ML infra: vendored-checkpoint verification + synthetic test model
  step 02  prompts/steps/job-04-step-02-thermo-ml.md  ThermoML estimator (CheMeleon) - replacement of RMG's
  step 03  prompts/steps/job-04-step-03-kinetics-ml.md  KineticsML estimator (Chemprop reactions)
  step 04  prompts/steps/job-04-step-04-registry.md  Rate registry: tunneling + forward/reverse wiring
  step 05  prompts/steps/job-04-step-05-estimation.md  estimation.py: library -> ML -> coverage error
  step 06  prompts/steps/job-04-step-06-gate.md  Thesis test: coverage + accuracy + no-fallback proof

## The job gate

Run by the final step's session (gates/gate_04.py, report
reports/job-04.md):

gates/gate_04.py - THE PoC THESIS TEST (PLAN.md 1, 10-Phase 1). Rigorous and honest: 1. Coverage: over the union of species in examples {superminimal, c3h4, minimal, minimal_ml, ethane-oxidation} + all species in primaryThermoLibrary: how many does the thermo model cover? (% + list of uncovered with reasons). Same for reactions (reactant->product pairs from the c3h4/superminimal mechanism-relevant set + the depository subset). 2. Accuracy vs RMG-Py: for every COVERED species (cap ~500, stratified): RMG-Py (library+GA, or ML per minimal_ml config where applicable) gives Hf298/S298/Cp(300,600,1000); compare to rmgpu ML values. Report: N, mean abs err, median, max, p95 for Hf298 (kJ/mol), S298 (J/mol/K), Cp (J/mol/K) + the distributions (histograms in the report). Same for HPL k(T): log10(k_rmgpu/k_rmg) stats at 300/600/1000 K for covered reactions. 3. No-fallback proof: the resolver NEVER falls back (instrumented: count library hits vs ML hits vs coverage errors; the report shows the split). 4. Checkpoint round-trip: BOTH vendored checkpoints (models/chemeleon_thermo_122e91.ckpt, models/chemprop_kinetics_122e91.ckpt) load via rmgpu.ml.base and reproduce the reference predictions recorded in the step-01 inventory (real checkpoints are present in the tree); the synthetic-chemprop models (step 1) still round-trip through predict() with exact values for the plumbing tests (clearly labeled: plumbing, not accuracy). A RED gate (errors above what makes a mechanism sensible, or coverage below what the test sets need) is a VALID PoC OUTCOME - reported as the thesis- test result with full numbers, recorded in STATUS, user decides. Do not paper over it.

## When the job is done

The final step's report (reports/job-04.md) has the gate result, the
job table row is `done` (or `blocked` with the cause), and the NEXT pointer
in STATUS.md targets job-05's first step (if there is a next job).
