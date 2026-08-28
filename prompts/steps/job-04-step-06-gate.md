# job-04/step-06: Thesis test: coverage + accuracy + no-fallback proof

Job: job-04 - ML estimators (thermo + kinetics) + rate registry + thesis test
Prereq: jobs 01-02 done (Molecule for structures; rate registry; the ml_estimator block of job 03)
This step is part of that job. The job's overall goal:
The SOLE property estimators, wrapping the two VENDORED checkpoints in this repo's `models/` directory (PLAN.md 3b): the CheMeleon thermo checkpoint (`models/chemeleon_thermo_662946.ckpt`, 9 log10-space targets: log_H298_J_mol, log_S298_J_mol_K, log_Cp_1..7_J_mol_K at 300/400/500/600/800/1000/1500 K) and the Chemprop RIGR reaction checkpoint (`models/chemprop_kinetics_662946.ckpt`, targets log10_A, n, Ea_J_mol; A = per-site pre-exponential in cm^3/(mol*s); input = atom-mapped reaction SMILES). No group additivity, no rate rules, no fallbacks - by design. Where a model does not cover a structure, the system records a COVERAGE GAP (a finding), it does not silently estimate. THE REPLACEMENT: RMG-Py's existing Chemprop-based estimators (rmgpy/ml/estimator.py) are NOT reused - read only for uncertainty-cutoff concepts and the ml_estimator DSL wiring. rmgpu's estimators are re-implemented wrapping the vendored checkpoints with the inference pattern of `models/predict.py` (the model team's own inference code, copied into this repo) - load checkpoint, featurize with the model's featurizer, data.build_dataloader, pl.Trainer(...).predict. LOAD-PATH CONSTRAINT: the checkpoints pickle-reference `models.BoundedOutputTransform` and `models.HuberMetric`, so a module importable as top-level `models` (i.e. `models/models.py`) is required to load them - do not rename/move it. Boundary conversions (raw -> SI, PLAN.md 3b): thermo = 10^pred; kinetics A = 10^pred * degeneracy (CGS), n as-is, Ea as-is. Deliverables: rmgpu/ml/ (base, thermo_estimator, kinetics_estimator), rmgpu/data/estimation.py (the only estimation code), and the thesis-test gate. The rate registry landed in job 02; this job wires ML output into it.
The job's gate (run by the job's final step):
gates/gate_04.py - THE PoC THESIS TEST (PLAN.md 1, 10-Phase 1). Rigorous and honest: 1. Coverage: over the union of species in examples {superminimal, c3h4, minimal, minimal_ml, ethane-oxidation} + all species in primaryThermoLibrary: how many does the thermo model cover? (% + list of uncovered with reasons). Same for reactions (reactant->product pairs from the c3h4/superminimal mechanism-relevant set + the depository subset). 2. Accuracy vs RMG-Py: for every COVERED species (cap ~500, stratified): RMG-Py (library+GA, or ML per minimal_ml config where applicable) gives Hf298/S298/Cp(300,600,1000); compare to rmgpu ML values. Report: N, mean abs err, median, max, p95 for Hf298 (kJ/mol), S298 (J/mol/K), Cp (J/mol/K) + the distributions (histograms in the report). Same for HPL k(T): log10(k_rmgpu/k_rmg) stats at 300/600/1000 K for covered reactions. 3. No-fallback proof: the resolver NEVER falls back (instrumented: count library hits vs ML hits vs coverage errors; the report shows the split). 4. Checkpoint round-trip: BOTH vendored checkpoints (models/chemeleon_thermo_662946.ckpt, models/chemprop_kinetics_662946.ckpt) load via rmgpu.ml.base and reproduce the reference predictions recorded in the step-01 inventory (real checkpoints are present in the tree); the synthetic-chemprop models (step 1) still round-trip through predict() with exact values for the plumbing tests (clearly labeled: plumbing, not accuracy). A RED gate (errors above what makes a mechanism sensible, or coverage below what the test sets need) is a VALID PoC OUTCOME - reported as the thesis- test result with full numbers, recorded in STATUS, user decides. Do not paper over it.

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

- Previous step (already done, its code is in the tree): job-04/step-05-estimation
- This step: job-04/step-06-gate
- Next step (do NOT start it): the job gate for job-04 (its step file)

## Goal

Run the job-04 gate - THE PoC THESIS TEST. This is the moment the
plan's premise (ML estimators match RMG's library+GA+rate-rules accuracy)
gets measured. Rigorous and honest: a red gate is a valid PoC outcome,
reported with full numbers.

## Reference to read (this step's budget)

  (no new reference reads - the gate consumes the estimators +
    RMG-Py as reference)
  RMG-Py/examples/rmg/{superminimal,c3h4,minimal,minimal_ml,
    ethane-oxidation}/input.py (the species/reaction sets)
  RMG-Py/rmgpy/rmg/main.py (2794 - READ THE THERMO/KINETICS ESTIMATION
    CALL SITES ONLY - how RMG-Py gets Hf298/S298/Cp and k(T) per
    species/reaction for the reference values (library+GA path; the ML path
    per minimal_ml config where applicable))

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- gates/gate_04.py: implements the job definition:
    (1) coverage over the example species union + primaryThermoLibrary
    species (thermo model) + the c3h4/superminimal mechanism-relevant
    reaction set (kinetics model) - % + uncovered list with reasons;
    (2) accuracy vs RMG-Py on covered species (cap ~500, stratified):
    Hf298/S298/Cp(300,600,1000) - N, mean abs err, median, max, p95 +
    histograms; HPL k(T): log10(k_rmgpu/k_rmg) at 300/600/1000 K;
    (3) no-fallback proof: the resolvers' counts (library/ML/coverage-error
    split) - assert the fallback count is 0;
    (4) checkpoint round-trip: BOTH vendored checkpoints load via
    rmgpu.ml.base and reproduce the step-01 reference predictions (the real
    checkpoints are in the tree - this is a plumbing assertion, not
    accuracy); the synthetic-chemprop models (step 1) still round-trip
    through predict() with exact values for the plumbing unit tests
    (clearly labeled: plumbing, not accuracy).
- The RMG-Py reference: a script (scripts/reference_props.py) that, in env
  rmg_env, loads the example mechanisms + libraries and dumps
  Hf298/S298/Cp(T) per species + k(T) per reaction to
  gates/baselines/thesis_test/ (commit the baselines).
- reports/job-04.md: checkpoint verification (from step 1 - both vendored
  checkpoints in models/ load + reference predictions), coverage policy
  text, the full thesis-test numbers (both estimators), the no-fallback
  split, and any coverage-gap findings (e.g. the positive-H298 training
  range vs rmgdb species with negative Hf298 - PLAN.md 3b).
- If RED (coverage/accuracy below what a mechanism needs): the report says
  so plainly with the numbers; STATUS gets a decisions-log entry; the user
  decides next steps. Do NOT paper over it.

## Checks (must run and pass before you claim done)

  /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_04.py
    -> PASS/RED recorded with the full numbers in reports/job-04.md
  pytest tests/ -q -> all pass

## Pitfalls

- This gate's result is a FINDING, not a bug to hide: the PoC's
  premise is measured here (PLAN.md 13 risk 1). The error distribution is
  the deliverable, not a smoothed "green".
- The stratified sample (cap 500) must be stratified by species class
  (radicals, intermediates, stable) - the coverage gap that matters most is
  intermediates (PLAN.md 13 risk 2).
- RMG-Py reference values: use its LIBRARY+GA path (the ML path only where
  the minimal_ml config applies) - the comparison is rmgpu-ML vs
  RMG-library+GA (that is the thesis: does ML match the legacy accuracy?).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-04/step-06: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-04/step-06 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-04-step-06-gate.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
