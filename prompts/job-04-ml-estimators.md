# job-04: ML estimators (thermo + kinetics) + rate registry

Read ORIENTATION.md (External: chemprop/CheMeleon; master equation data path) and
PLAN.md sections 3 (deleted subsystems - the ones these replace), 5 (delegations),
6 (chemprop reaction prediction, verified API), 1 (the pipeline diagram). Prereqs:
jobs 01-02.

## Goal

The SOLE property estimators: CheMeleon/Chemprop for thermo (Hf298, S298, Cp(T)) and
Chemprop reaction models for high-pressure-limit kinetics. No group additivity, no
rate rules, no fallbacks - by design. Where a model does not cover a structure, the
system records a COVERAGE GAP (a finding), it does not silently estimate.

## Reference (read)

  RMG-Py/rmgpy/ml/estimator.py  -- RMG's existing (disabled) chemprop wrapper for
                                   thermo: MLEstimator(hf298_path, s298_cp_path),
                                   uncertainty cutoffs. Read it in full; it is the
                                   seam. It is species-only; kinetics is NEW.
  RMG-Py/examples/rmg/minimal_ml/input.py  -- how the ml_estimator DSL is wired.
  Checkpoint locations: see reports/job-00.md (found in job 00).
  chemprop API (installed in env; verify against installed version):
    - species: ChempropModel + featurizer (MolGraphFeaturizer); predict -> array
    - reactions: Rxn = tuple[Mol, Mol]; RxnMode enum (REAC_PROD, REAC_PROD_BALANCE,
      REAC_DIFF, REAC_DIFF_BALANCE, PROD_DIFF, PROD_DIFF_BALANCE);
      reaction featurizers in chemprop/featurizers/molgraph/reaction.py;
      training/prediction threading via rxn_mode (verified in plan, PLAN.md 6).

## Deliverables

1. `rmgpu/ml/thermo_estimator.py`
   - `ThermoML(models_dir)` loads the CheMeleon checkpoints (Hf298 model; S298+Cp
     model - match RMG's ml/estimator.py layout).
   - `predict(molecule) -> (Hf298, S298, Cp_model, uncertainties)` in SI.
     Cp model: whatever the checkpoint predicts (T-grid or coefficients - inspect
     the checkpoints in job 00 and document the exact output format in the report;
     wire a `CpModel` that converts to the Wilhoit-compatible Cp(T) representation
     in rmgpu.data.thermo so the rest of the code sees one thermo model).
   - coverage policy: the estimator exposes `.covers(molecule) -> bool` (heuristic:
     element set / heavy-atom count / SMILES parse; document the exact policy).
     If not covered: raise `MLCoverageError` (captured by callers, recorded as a
     finding; NEVER a silent fallback).
   - batch mode: predict on a list (torch .to(device) batching; device = cuda if
     available else cpu - single GPU task rule applies).
2. `rmgpu/ml/kinetics_estimator.py`
   - `KineticsML(models_dir)` loads a Chemprop REACTION model (checkpoints dir; if
     none exists yet on this box, the module must be written + unit-tested against
     a SYNTHETIC chemprop model trained in the test (small mol set, random target)
     to prove the plumbing, and the real-checkpoint path is documented as
     BLOCKED-until-checkpoint in the report - see gate note).
   - `predict(reaction: (reactant_mol, product_mol)) -> KineticsPrediction` where
     KineticsPrediction is either Arrhenius parameters (A, n, Ea) or a k(T) grid -
     whatever the model outputs (inspect checkpoint metadata; document). RxnMode is
     a constructor argument (default REAC_DIFF; record the choice + why in the
     report).
   - same coverage policy as thermo (reactant+product both covered).
3. `rmgpu/kinetics/models.py` -- the rate-expression registry (numpy/torch, thin):
   - `RateModel` union: Arrhenius, ArrheniusEP, Falloff (Lindemann/Troe/ThirdBody +
     high-pressure-limit model + low-pressure-limit), Chebyshev/PDepArrhenius
     (pdep fits, job 07 produces these), Marcus, Wigner/Eckart tunneling
     (port the math from RMG-Py/rmgpy/kinetics/tunneling.pyx - Wigner at minimum).
   - `k(T, P)` evaluation in SI; `.forward_reverse()` thermodynamic-consistency
     helper (port generate_reverse_rate_coefficient from kinetics/model.pyx - this
     uses dG(T) from the thermo models, so it ties jobs 02/04 together).
   - This registry WRAPS ML output: the estimator returns params, the registry is
     what the reactor and master equation consume.
4. `rmgpu/data/estimation.py` -- the two resolver functions (the ONLY estimation
   code in the package):
   - `estimate_thermo(species, databases, ml)`: library hit -> library value; else
     ML; else MLCoverageError (recorded).
   - `estimate_kinetics(reaction, databases, ml)`: library hit -> library; else ML;
     else MLCoverageError (recorded).
   These replace the multi-method fallback chains (PLAN.md section 3).
5. `reports/job-04.md` MUST include: checkpoint inventory (paths, formats, output
   dims), the coverage policy text, and the error analysis (below).

## Gate (job 04) -> gates/gate_04.py, report reports/job-04.md

THIS IS THE PoC THESIS TEST (PLAN.md 1, 10-Phase1). It must be rigorous and honest.

  1. Coverage: over the union of species in examples {superminimal, c3h4,
     minimal, minimal_ml, ethane-oxidation} + all species appearing in the
     primaryThermoLibrary: how many does the thermo model cover? (%, list of
     uncovered with reasons). Same for reactions in the c3h4/superminimal
     mechanism-relevant set (reactant->product pairs from RMG-Py rate rules
     training depository subset + generated pairs).
  2. Accuracy vs RMG-Py: for every COVERED species (cap at ~500, stratified sample
     if more): run RMG-Py (its library+GA+ML estimator, per the minimal_ml config
     where applicable, GA otherwise) to get Hf298/S298/Cp(300,600,1000); compare to
     rmgpu ML values. Report: N tested, mean abs err, median, max, p95 for Hf298
     (kJ/mol) and S298 (J/mol/K) and Cp (J/mol/K), and the distribution (histogram
     in the report). Same for HPL k(T): log10(k_rmgpu/k_rmg) stats at 300/600/1000
     K for the covered reactions.
  3. No-fallback proof: assert the resolver NEVER falls back (instrument it; count
     library hits vs ML hits vs coverage errors; the report shows the split).
  4. Synthetic-model plumbing test (if real kinetics checkpoint absent): the
     synthetic-chemprop reaction model round-trips through predict() with exact
     values (proves the code path works; clearly labeled as plumbing, not accuracy).
  A RED gate here (ML errors above what makes a mechanism sensible, or coverage
  below what the test sets need) is a VALID PoC OUTCOME - it is reported as the
  thesis-test result with full numbers, recorded in STATUS.md, and the user decides
  next steps. Do not paper over it.

## When done

STATUS.md (job 04 row, + a decisions-log entry summarizing the thesis-test numbers
in one paragraph) + commit "job-04: ML estimators + rate registry + thesis-test
report" + STOP.

## Pitfalls

- CheMeleon checkpoint format: inspect before assuming (job 00 started this; finish
  it). The estimator must be checkpoint-agnostic enough that a new checkpoint
  (better model) is a config change, not code (the "re-point the checkpoint"
  decision, PLAN.md 8a.3).
- Units: checkpoints predict in chemprop's training units (kJ/mol, J/mol/K, or
  kcal - verify per checkpoint). Convert at the estimator boundary; SI inside.
- Do not import rmgpy anywhere in rmgpu (the gate scripts may, for reference).
- Keep the estimators device-managed: one torch device, no data loading that thrashes
  GPU memory; batch sizes small (this box is shared with llama-server).
