# job-04/step-03: KineticsML estimator (Chemprop reactions)

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

- Previous step (already done, its code is in the tree): job-04/step-02-thermo-ml
- This step: job-04/step-03-kinetics-ml
- Next step (do NOT start it): job-04/step-04-registry

## Goal

Build rmgpu's kinetics ML estimator: Chemprop REACTION models
(reaction mode: the RxnMode enum) predicting HPL rates. This is NEW in the
design - RMG's MLEstimator was species-only; the reaction generalization is
exactly what chemprop natively supports (PLAN.md 6, verified in the 2.3.1
source).

## Reference to read (this step's budget)

  chemprop reaction featurizers (installed; the API):
    chemprop/featurizers/molgraph/reaction.py - RxnMode enum + the reaction
    featurizer (which class takes (rct, pdt) mols + rxn_mode)
  chemprop/types.py: Rxn = tuple[Mol, Mol]
  chemprop/data/datapoints.py: ReactionDatapoint /
    LazyReactionDatapoint (rct_smiles / pdt_smiles)
  (the synthetic reaction model from step 1; base.py)
  RMG-Py/rmgpy/ml/estimator.py  (182 - the species-only OLD estimator: the
    reaction path has NO counterpart in RMG - this is net new, informed by
    PLAN.md 6's API notes)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/ml/kinetics_estimator.py:
    `KineticsML(models_dir, rxn_mode=RxnMode.REAC_DIFF)` - loads a Chemprop
    REACTION model via base.load_chemprop_model; RxnMode is a constructor
    argument (default REAC_DIFF; the choice + why is recorded in the
    report - the rate rules' feature basis is the guide: which mode's graph
    combination best matches it; PLAN.md 6). If the real reaction
    checkpoint does not exist on the box yet: the module is written +
    unit-tested against the SYNTHETIC reaction model (step 1), and the
    real-checkpoint path is documented as BLOCKED-until-checkpoint in the
    report (the gate's note 4 covers this).
    `predict(reaction) -> KineticsPrediction` where reaction =
    (reactant_mol, product_mol) (rct/pdt; atom-mapped where the mode needs
    it - document what mapping the featurizer expects); output = Arrhenius
    params (A, n, Ea) or a k(T) grid - whatever the model outputs (inspect
    the checkpoint metadata; document).
    Coverage: reactant AND product both covered (reuse the thermo policy
    per-molecule).
- tests/test_kinetics_ml.py: synthetic reaction model round-trips through
  predict() with exact values (plumbing - labeled as such); RxnMode
  dispatch (predict works with each mode on the synthetic model); coverage
  cases (one side uncovered -> MLCoverageError).

## Checks (must run and pass before you claim done)

  pytest tests/test_kinetics_ml.py -q -> all pass
  the synthetic reaction checkpoint (step 1) loads + predicts via
    KineticsML (values exact)

## Pitfalls

- Atom mapping: the reaction featurizer may need atom-mapped
  reactant/product (which atom maps to which) - check the featurizer's
  expected input and document it; RMG's recipe engine (job 05) produces the
  atom correspondence, but for this step a simple canonical mapping suffices
  for the synthetic tests.
- The output format (A/n/Ea vs k(T) grid) is per-checkpoint - the class
  must document what its checkpoint predicts and convert to the registry's
  expectation (Arrhenius params preferred; the registry wraps, job 04's
  estimation.py consumes).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-04/step-03: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-04/step-03 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-04-step-03-kinetics-ml.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
