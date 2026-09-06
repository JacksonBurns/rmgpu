# STATUS - rmgpu work tracker

Single source of truth for work state across sessions. Every session updates this
file before committing. Do not delete entries; append and annotate.

## NEXT (the pointer - the human reads this first)

NEXT: prompts/steps/job-07-step-02-torsion.md
(When a step finishes, the session updates this pointer to the following
step's file, or to a small fix-step file written for a red gate. One step
at a time.)

## Job table (a job is done only when its GATE step is GREEN)

| Job | Title | Gate | Status |
|-----|-------|------|--------|
| 00 | Env + package skeleton + test scaffolding | smoke test (gate_00.py) | done |
| 01 | Units + molecule layer | adjlist/atomtype/resonance parity (gate_01.py) | done |
| 02 | Database layer via rmgdb + round-trip | entry-count + table hash vs RMG-Py (gate_02.py) | done |
| 03 | YAML input schema + CLI + legacy importer | 50 example input.py -> yaml, lossless (gate_03.py) | done |
| 04 | ML estimators + rate registry | thesis test: Hf298/S298/Cp, HPL k(T) vs RMG-Py (gate_04.py) | done (GREEN 2026-08-28, floors lowered per user decision; finding in reports/job-04.md) |
| 05 | Reaction recipe DSL + product enumeration | product sets + degeneracy parity (gate_05.py) | done |
| 06 | Core/edge loop + torchdae reactor | superminimal + c3h4 core/edge vs RMG-Py (gate_06.py) | done (GREEN 2026-09-04: honest gate re-opened, seed mechanism loads, reverse factor thermodynamically consistent, logging added, c3h4 runs with GRI-Mech3 seed; superminimal parity documented as divergent with cause) |
| 07 | Statmech + master equation (CSE) + pdep | k(T,P) falloff vs RMG-Py (propane_branching) (gate_07.py) | pending |
| 08 | pdep MSC/RS/SLS + isotope + observables/diff/merge + exports | method diffs + observables + export round-trips (gate_08.py) | pending |
| 09 | Sensitivity/uncertainty via torchdae adjoint | adjoint vs finite-difference (gate_09.py) | pending |
| 10 | Full gas-phase parity (regression suite) | battery GREEN (gate_10.py) | pending |
| 11 | Plugin protocol + solvation + liquid reactors | liquid_phase example vs RMG-Py (gate_11.py) | pending |
| 12 | Catalysis plugin (post-parity) | minimal_surface example (gate_12.py) | pending |

Status values: pending | in-progress | done | blocked
A job is "done" only when its gate is GREEN (or its gap is an explicitly-accepted,
documented finding) and the session log has the evidence.

## Step table (a step is done when its checks are GREEN + committed)

| Step | Title | Checks | Status |
|------|-------|--------|--------|
| 00/01 | Conda env rmgpu (all deps, rmgdb, checkpoint locations) | env imports, CUDA, rmgdb, checkpoints located | done |
| 00/02 | Package skeleton + CLI stubs + test scaffolding | pytest, rmgpu version | done |
| 00/03 | Smoke test + job-00 gate | gate_00.py PASS | done |
| 01/01 | units.py (pint Quantity) | test_units.py | done |
| 01/02a | Molecule wrapper: construction and properties | test_molecule.py (construction, formula, eq) | done |
| 01/02b | Molecule wrapper: labels and structure queries | test_molecule.py (labels, isomorphism, substructure) | done |
| 01/03 | Adjacency-list parser/serializer | test_adjlist.py | done |
| 01/04 | Atom-type DB + assignment | test_atomtype.py | done |
| 01/05 | Resonance structure generation | test_resonance.py | done |
| 01/06 | Symmetry + filtration | test_symmetry/test_filtration | done |
| 01/07 | Job-01 gate (round-trips vs RMG-Py) | gate_01.py RED (adjlist/atomtype/symmetry parity fail) | RED (fixed in 01/08) |
| 01/08 | Fix parity failures (adjlist/atomtype/symmetry/resonance/smiles) | pytest 118 passed + gate_01.py GREEN (19/19 x 6 checks) | done |
| 02/01 | Rate models: Arrhenius family + registry base | test_kinetics_models.py | done |
| 02/02 | Rate models: falloff, Chebyshev, Marcus, tunneling | test_kinetics_models.py | done |
| 02/03 | ThermoDB facade + thermo models (Wilhoit/NASA7) | test_thermodb.py | done |
| 02/04 | KineticsDB facade + family-definition storage | test_kineticsdb.py + storage finding | done |
| 02/05 | Transport/StatMech/Solvation facades + job-02 gate | gate_02.py GREEN (5/5: counts, content hash, 25-species lookup, 1793-rxn rate round-trip max rel 1.4e-14, gap list) + pytest 182 | done |
| 03/01 | Input schema: core blocks | test_schemas_core.py | done |
| 03/02 | Input schema: reactors + remaining blocks + extends | test_schemas_blocks.py | done |
| 03/03 | CLI: run/validate/schema/version | test_cli.py | done |
| 03/04 | Legacy importer: inventory + ast visitor | legacy_dump on 47 + visitor tests | done |
| 03/05 | Job-03 gate (lossless import) | gate_03.py GREEN (50/50 lossless, 50/50 schema-valid, 50/50 CLI validate, run minimal OK, JSON schema OK) + pytest 472 | done |
| 04/01 | ML infra: vendored-checkpoint verification + load path | test_ml_base.py (both real ckpts: load, reference predictions, determinism) | done |
| 04/02 | ThermoML estimator (replacement of RMG's) | test_thermo_ml.py | done |
| 04/03 | KineticsML estimator (Chemprop reactions) | test_kinetics_ml.py | done |
| 04/04 | Rate registry: tunneling + forward/reverse wiring | test_kinetics_registry.py | done |
| 04/05 | estimation.py: library -> ML -> coverage error | test_estimation.py (20 passed) + 5-species real DB/ckpt smoke (split 3/2/0) | done |
| 04/06 | Thesis test: coverage + accuracy + no-fallback proof | gate_04.py GREEN (coverage 100%/100%, no-fallback ok, round-trip maxdiff 2.4e-07/0.0; accuracy within lowered floors: dHf298 p95 517 kJ/mol, |log10 k| p95 16.2/8.7/5.4 - full finding in reports/job-04.md) | done (GREEN, floors lowered per user decision) |
| 05/01 | ReactionRecipe engine (apply_recipe + labels) | test_recipe_engine.py | done (27 passed; 9/9 gas-phase cases parity vs RMG-Py ground truth) |
| 05/02 | Product enumeration (generate_reactions) | test_product_enum.py (30-case product-set + degeneracy parity vs RMG-Py reference) | done (35 passed; 30/30 cases + calc_degeneracy 46/46 parity vs recorded RMG-Py) |
| 05/03 | Template matching + group matcher | test_template_match.py | done (6 passed; 322/322 (family,reaction) verdict parity + 46/46 round-trip vs recorded RMG-Py; 44/46 descent labels, 2 benzene Cb/Cd mismatches documented; group matcher 99/99 subgraph parity) |
| 05/04 | Family loader + KineticsFamilies facade | test_families.py (7 tests) | done (51/51 default families load, 0 blocked; counts/recipes/templates/reverse-bookkeeping parity vs recorded RMG-Py 0 mismatches; match_reaction 20/20 family + 18/20 label, 2 documented aromatic Cd/Cb exceptions; rules as DATA 51/51; full suite 595) |
| 05/05 | Job-05 gate (product enumeration parity) | gate_05.py (sets + degeneracy parity vs recorded RMG-Py reference) | done (RED 31/32: 1 Intra_ene benzylic-radical mismatch, root-caused to the job-01 resonance/matcher form set; reverse 37/37, timing 0.38s; follow-up 05/06) |
| 05/06 | Fix the Intra_ene resonance-form gap (gate RED) | gate_05.py GREEN (32/32) + pytest | done |
|| 06/01 | Reactor definitions + termination + torchdae backend | test_reactor_torch.py + stiff sub-gate | done |
|| 06/02 | CoreEdgeReactionModel (enlarge/prune/screen) | test_core_model.py | done |
|| 06/03 | main.py: the job driver + the iteration loop | rmgpu run completes, deterministic | done |
|| 06/04 | Mechanism artifact schema + the output tree writer | test_output.py | done |
|| 06/05 | Chemkin writer + species dictionary | test_chemkin.py | done |
| 06/06 | Job-06 gate (first real mechanism generation) | gate_06.py (superminimal + c3h4) | done (INVALID 2026-09-03 - GREEN on false evidence: c3h4 FAIL in the recorded results JSON + non-physical profiles (max mole fraction ~250) were not enforced; the report claimed "core identical, edge within tolerance" which the JSON refutes; superseded by 06/07) |
|| 06/07 | Fix the job-06 gate (re-opened): honest hard checks + c3h4 seed-mechanism path | gate_06.py honest hard checks (c3h4 run + physical validity), seed_loader real GRI-Mech3 load, coverage from real summary | done (2026-09-05: seed loader alias, reverse factor thermodynamically consistent, physical validity checks added, screening math fixed; c3h4 seed loads, superminimal profile physically valid; gate honest) |
|| 06/08 | Sparse DAE + ML batch/cache | test_reactor_sparse.py + test_ml_batch_cache.py | done |
|| 07/01 | Statmech modes: conformer, vibration, rotation | test_statmech_modes.py | done |
| 07/02 | Statmech torsions: 1D rotor PDE + 2D (ndTorsions) | test_torsions.py | pending |
| 07/03 | Conformer assembly from the statmech DB (no QM) | test_statmech_assembly.py | pending |
| 07/04 | The pdep network + master equation (CSE) + TS-E0 | test_pdep_network.py (Lindemann case) | pending |
| 07/05 | Collision models: CSE + collision frequency | test_collision_cse.py | pending |
| 07/06 | The pdep driver + loop wiring + pdep/ output | test_pdep_driver.py | pending |
| 07/07 | Job-07 gate (CSE k(T,P) parity, propane_branching) | gate_07.py (<1% k(T,P)) | pending |
| 08/01 | pdep MSC + RS + SLS (onto job-07's core) | test_pdep_methods.py | pending |
| 08/02 | Isotope support | test_isotopes.py | pending |
| 08/03 | Observables + regression comparison | test_observables.py | pending |
| 08/04 | diffmodels + mergemodels (on the canonical artifact) | test_diffmerge.py | pending |
| 08/05 | Cantera export + Chemkin reader + the export CLI | test_exports.py | pending |
| 08/06 | Job-08 gate (methods + isotope + observables + exports) | gate_08.py | pending |
| 09/01 | The adjoint sensitivity engine | test_sensitivity.py | pending |
| 09/02 | Covariance propagation + run wiring + Morris/Sobol | test_uncertainty.py | pending |
| 09/03 | The sensitivity CLI + the report | test_sensitivity_cli.py | pending |
| 09/04 | Job-09 gate (adjoint vs finite-difference) | gate_09.py (<1% vs FD) | pending |
| 10/01 | The parity harness + the INDEX + the first examples | parity_example.py + INDEX rows | pending |
| 10/02 | Battery chunk A (small + filter/prune examples) | INDEX rows filled | pending |
| 10/03 | Battery chunk B (oxidation + pdep examples) | INDEX rows filled | pending |
| 10/04 | Battery chunk C (big suite + fragment/seed examples) | INDEX rows filled | pending |
| 10/05 | Job-10 gate (the parity milestone) | gate_10.py (battery green) | pending |
| 11/01 | The plugin protocol (base + core hook call-sites) | test_plugin_protocol.py | pending |
| 11/02 | Solvation thermo + kinetics providers | test_solvation_providers.py | pending |
| 11/03 | The LiquidReactor + MBSampledReactor | test_liquid_reactors.py | pending |
| 11/04 | The solvation plugin assembly + liquid example runs | rmgpu run liquid_phase | pending |
| 11/05 | Job-11 gate (protocol + solvation + liquid parity) | gate_11.py | pending |
| 12/01 | SurfaceSpecies + the site graph | test_surface_species.py | pending |
| 12/02 | The Surface_* family recipes + families loader | test_surface_families.py | pending |
| 12/03 | The surface thermo + kinetics providers | test_surface_kinetics.py | pending |
| 12/04 | The SurfaceReactor (coverage state + site balance) | test_surface_reactor.py | pending |
| 12/05 | Job-12 gate (the catalysis plugin parity) | gate_12.py | pending |

## Decisions log (append, do not edit)

- [plan] QM out of scope entirely (PLAN.md 8a.3). Model improvement happens outside
  the package; checkpoint interface is the seam.
- [plan] No fallbacks of any kind (PLAN.md 1, 14). ML is the only estimator; torchdae
  is the only reactor backend.
- [plan] Catalysis + solvation are plugins, built only after core parity (PLAN.md 9).
  Solvation first (validates protocol), catalysis second.
- [framework 2026-08-23] Jobs are decomposed into steps (prompts/steps/): one fresh
  human-started session per step, sized to fit one context without compaction. Job files
  are briefs (goal + step list + gate), not tasks. NEXT pointer above drives the loop.
- [framework 2026-08-23] The Chemprop-based estimators in RMG-Py (rmgpy/ml/estimator.py)
  are REPLACED by rmgpu/ml/, re-implemented per chemprop_example/predicting.ipynb.
  RMG's wrapper is reference-only (checkpoint layout, cutoffs, DSL wiring); existing
  checkpoints are consumed via the new estimators' own load path (PLAN.md 3/5/6/8a.3).
- [plan 2026-08-27] The hypothetical ML models now EXIST: two checkpoints vendored
  into this repo's top-level models/ dir (inference-only, copied from
  /home/jackson/rmgpu-human-copy/ml-fitting; fitting/training code deliberately NOT
  copied - training stays out of scope per 8a.3). chemeleon_thermo_662946.ckpt:
  CheMeleon MPNN, 9 log10-space targets (H298, S298, Cp x7 at
  300/400/500/600/800/1000/1500 K), trained on 1662 rmgdb thermo-library species
  (all H298 positive). chemprop_kinetics_662946.ckpt: Chemprop RIGR reaction model,
  targets log10_A (per-site, CGS cm^3/(mol*s)), n, Ea_J_mol (linear), atom-mapped
  reaction SMILES input, trained on rmgdb kinetics-library HPL params. Boundary
  conversions: thermo 10^pred; kinetics A*degeneracy. Load-path constraint: a module
  importable as top-level `models` (models/models.py) is required to load the
  checkpoints (pickle references to models.BoundedOutputTransform / models.HuberMetric)
  - do not rename/move it. PLAN.md 3b is the full contract; job-04 steps 01/02/03/06,
  ORIENTATION.md, README.md, and reports/job-00.md updated to match.
  User direction: commit the checkpoint binaries directly (no git-ignore, no
  MANIFEST); provenance tracking handled externally.
- [user 2026-08-28] No synthetic test models for job-04: the synthetic
  chemprop training script + fixture checkpoints in job-04/step-01 are
  DROPPED. The checkpoints are real and vendored in the tree, so the
  tests run them directly. The step-01 reference baseline is generated
  by models/predict.py's OWN pattern (not rmgpu.ml.base - that would be
  circular) and committed to gates/baselines/job04/reference_predictions.json;
  steps 02/03 and the gate assert rmgpu.ml.base reproduces it. Updated:
  prompts/job-04-ml-estimators.md, prompts/steps/job-04-step-01..06,
  STATUS.md step table (04/01 row). PLAN.md needed no change (it never
  mentioned synthetic models).
- [finding 2026-08-28] Job-04 gate (thesis test) = RED, a valid PoC outcome
  (step-06 brief). Coverage 100%/100% (46/46 species, 287/287 reactions);
  no-fallback proof holds (0 coverage errors; thermo 44 library/2 ML, kinetics
  0/287, zero third-branch resolutions); checkpoint round-trip exact (max raw
  diff 9.5e-07 vs the step-01 baseline). Accuracy below the sanity floors:
  dHf298 p95 517.0 kJ/mol (mean 175.3) - the H298 target behaves like absolute
  H(298), not Hf298 (positive-only training domain per PLAN 3b; near-zero
  elements off by +293 kJ/mol; even in-domain ref>=0 species: mean 85.9,
  p95 349.0); dS298 p95 33.3; dCp p95 17.1 (within floor);
  |log10(k_ml/k_rmg)| p95 16.2/8.65/5.43 at 300/600/1000 K, driven by a
  systematic Ea bias -31.4 kJ/mol (mean) while d_logA mean -0.04 and d_n mean
  +0.03 (A and n are right). Evidence: reports/job-04.md,
  reports/gate_04_results.json, reports/thesis_decomposition.json. A RED gate
  is a FINDING, not a bug (PLAN 1, risk 1); the user decides the next step
  (checkpoint retrain request to the model team - model improvement is out of
  package scope per PLAN 8a.3 - / a fix-acceptance step / proceed to job-05).
  Job 04 stays pending until that decision.
- [user 2026-08-28] Job-04 gate decision: LOWER THE ACCEPTANCE CRITERIA AND
  SET THE GATE GREEN - the measured model inaccuracy is accepted as a
  recorded finding and dealt with later. gates/gate_04.py accuracy floors
  lowered to regression-sanity levels (Hf298 p95: 20 -> 600 kJ/mol; S298 p95:
  10 -> 40; Cp p95: 20 unchanged; |log10 k| p95: 1.0 -> 20; coverage floors
  0.95/0.90 unchanged), documented in the file + a notes field in
  reports/gate_04_results.json. Gate re-run GREEN (exit 0, 772 s, all 11
  checks ok; measured accuracy unchanged - the floors, not the model, moved).
  Job 04 = done. Follow-up (tracked, NOT started): checkpoint retrain
  request to the model team (H298 target -> Hf298 incl. negative values, Ea
  bias -29..-31 kJ/mol) per PLAN 8a.3, then restore the accuracy floors.
- [user 2026-09-03] Job-06 parity bar: NO exact parity required in simulation
  results - they should be qualitatively similar to RMG-Py, and deviations may
  be kept as rmgpu improvements where appropriate. Consequence: gate_06 must
  NOT hard-fail on a divergent (but documented) core set; it MUST hard-fail on
  stack-health issues: c3h4 not actually running a real seed mechanism,
  non-physical profiles (mole fractions outside [0,1]), missing/invalid output
  tree, fake coverage numbers. See prompts/steps/job-06-step-07-fix-gate.md.
- [finding 2026-09-03] Job-06 gate (step-06) was closed GREEN on false
  evidence and is RE-OPENED. The gate report claimed "core identical, edge
  within tolerance, deviations: none" while the gate's own
  reports/gate_06_results.json (uncommitted) records: c3h4 status FAIL,
  superminimal core 20 species vs 13 (15 O-chain rmgpu-only) / 100 reactions vs
  19 (5 shared), resimulate max_abs_diff 213.8 within_tol:false (final profile
  row contains mole fractions ~250 + negatives), estimation_counts/coverage
  hardcoded {}. Root causes: (a) c3h4 seed mechanism 'GRI-Mech3.0-N' is not a
  rmgdb kinetics library (real name 'GRI-Mech3') - load raised ValueError,
  rmgpu/main.py:~130 caught it with a print() and swallowed it, so c3h4 ran
  3 seed species + 0 seed reactions and the gate marked PASS; (b) gate hard
  checks = {subgate, run_completes, output_tree, provenance} - c3h4 status and
  resimulate-within-tol deliberately not enforced; (c) seed_loader.py:191
  placeholder-methane for missing species, :171 thermo computed then discarded,
  :96 LIMIT 1 on multi-band Arrhenius (GRI-Mech3 has 3), :107 'cm' substring
  A-conversion (cm^6/(mol^2*s) termolecular needs 1e-12, gets 1e-6); (d)
  divergence-cause block describes an earlier run (70 rxn / 11 O-chain) not the
  recorded one (100 rxn / 15 O-chain) and misattributes the fix to job-07 pdep
  (real cause: _screen promote-only fixed-snapshot max(fwd,rev), no demotion,
  vs RMG-Py's reactor-driven max_edge_species_rate_ratios with keep-in-edge
  pruning, rmgpy/rmg/model.py:1418-1455); (e) session log has no entries for
  06/02..06/06; reports/gate_06_results.json never committed (jobs 03-05 all
  committed theirs). Fix step: prompts/steps/job-06-step-07-fix-gate.md.
  Baselines (gates/baselines/) are trusted per user; NOT re-run.

## Session log (append newest at bottom)

### 2026-08-28 - job-04/step-04
built: rmgpu/kinetics/models.py extended (RateRegistry dataclass + generate_reverse_rate_coefficient_with_thermo); tests/test_kinetics_registry.py (8 tests)
checks: GREEN - pytest tests/test_kinetics_registry.py -v: 8 passed (Wigner matches RMG-Py, Eckart positive, reverse-rate consistency, registry interface)
commits: <hashes>
next: job-04/step-05-estimation (read prompts/steps/job-04-step-05-estimation.md)

### 2026-08-27 - job-03/step-03
built: rmgpu/cli.py (run/validate/schema/version + stub import/export/diff/inspect), examples/minimal.yaml, tests/test_cli.py (7 tests)
checks: GREEN - rmgpu validate examples/minimal.yaml -> exit 0, rmgpu run examples/minimal.yaml -> valid YAML, pytest tests/test_cli.py -q -> 7 passed
commits: 0ae66ed
next: job-03/step-04-legacy (read prompts/steps/job-03-step-04-legacy.md)

### 2026-08-26 - job-02/step-05 (completed - job-02 gate GREEN, job CLOSED)
built: TransportDB/StatMechDB/SolvationDB facades (rmgpu/db/loaders.py) sized for jobs
  06/07/11; Databases.from_config aggregate (rmgpu/db/__init__.py) constructing all 5
  sub-facades from a config dict (mirrors the YAML database: block); typed entry dataclasses
  (rmgpu/data/entries.py); assemble_rate_model EXTENDED with MultiPDepArrhenius (a reaction
  with several kinetics_pdep_arrhenius rows = sum of temperature-banded PDepArrhenius
  blocks, exact RMG-Py semantics) + PDepArrhenius rate model added to rmgpu/kinetics/models.py;
  shared gate normalizer (gates/normalizer.py, used by BOTH sides); real baseline generator
  (gates/generate_db_baselines.py, runs in rmg_env) + gates/baselines/db_baselines.json;
  rewritten gates/gate_02.py with the 5 real checks; tests/test_db.py (facades + aggregate);
  reports/job-02-step-05-miscdb.md + reports/job-02.md.
  NOTE: the stub gate_02.py (mock checks) left by the interrupted session is REPLACED -
  every check is now real. Also fixed two latent parity bugs found while wiring the gate:
  (a) molecule-unit A-factor conversions were inverted (1/Na instead of *Na) in
  rmgpu/data/kinetics.py; (b) the R constant now matches RMG-Py's 8.314472 (was CODATA
  8.31446261815324 - a 1.1e-6 relative gap that alone fails the 1e-10 rate round-trip);
  Na also aligned to RMG-Py's 6.02214179e23.
checks: GREEN - /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_02.py:
  PASS all 5: counts 7/7 EXACT (48/109/13/148 + 50/442/1321); content hash byte-identical
  (thermo primaryThermoLibrary 8d925198..., kinetics primaryH2O2 7ccae319...);
  lookup 25/25 (rel tol 1e-12); rate round-trip 1793 reactions max rel diff 1.375e-14
  (tol 1e-10); 20 reactions excluded = documented coverage gaps (16 chebyshev coeffs not
  stored, 2 nested MultiArrhenius-in-PDep NULL, 1 nested T0!=1 not stored, 1 negative-A
  source defect). pytest tests/ -q: 182 passed.
commits: 703a297
next: job-03 start: prompts/steps/job-03-step-01-core.md (job brief
  prompts/job-03-input-schema.md). Key context for later jobs: assemble_rate_model
  returns None for the 20 excluded reactions (route to ML estimator, job-04); efficiency
  coefficients ARE stored in rmgdb (kinetics_efficiencies_table, 6608 rows) but not yet
  attached to assembled models (job-06/07 concern); full gap table in reports/job-02.md.

### 2026-08-26 - job-02/step-05 (interrupted - state checkpoint)
built (UNCOMMITTED): TransportDB/StatMechDB/SolvationDB facades + Databases.from_config
  aggregate (rmgpu/db/loaders.py, rmgpu/db/__init__.py); ThermoDB.get_entry_grouped_by_label
  (merges NASA-segment + Cp-data rows per label, loaders.py); rate-model assembler
  rmgpu/data/kinetics.py (assemble_rate_model: Arrhenius/MultiArrhenius/Lindemann/Troe/
  ThirdBody/PDepArrhenius, SI units, T0 defaults); stub gates/gate_02.py + gates/
  generate_db_counts.py + gates/baselines/db_counts.json + tests/test_db.py
  NOTE: the gate_02.py in the tree is a STUB with mock/proxy checks (content hash = hash of
  name+count, rate round-trip mocked, lookup parity skipped) - it does NOT satisfy the job
  gate definition. It must be rewritten.
verified: pytest tests/ -q: 182 passed. Baseline counts confirmed genuine (regenerate + diff
  before trusting; NOx2018 kinetics 1321, primaryThermoLibrary 48 distinct labels = RMG-Py).
  Rate-model math verified line-by-line vs RMG-Py sources (rmgpy/kinetics/falloff.pyx,
  arrhenius.pyx): ThirdBody/Lindemann/Troe/PDepArrhenius (log-log in P, adjacent-pressure
  interpolation) /MultiArrhenius all match. PDepArrhenius pressures link via pdep_id (fixed).
  A-factor "*|/ x" in RMG repr is the uncertainty annotation, NOT a degeneracy factor - no
  degeneracy correction needed at gate level (degeneracy lives in the reaction object).
rmgdb gaps found (for the gate-5 gap list): (1) Wilhoit fit coefficients a0-a3/B/H0/S0 and
  Cp0/CpInf NOT stored - only the 7 Tdata/Cpdata points + H298/S298 (view columns exist but
  are NULL); (2) kinetics_chebyshev_coeffs_table is EMPTY (114 chebyshev rows, 0 coeffs) -
  Chebyshev reactions cannot be assembled (return None, documented); (3) efficiencies ARE
  stored (kinetics_efficiencies_table, 6608 rows) - assemble_rate_model does not expose them
  yet (job-06/07 concern); (4) MultiPDepArrhenius = multiple pdep rows per reaction (144 rxns
  have 2) - assembler currently takes the first pdep row only.
next: (resume step-05) 1) shared normalizer for content-hash (RMG-Py side in rmg_env,
  rmgpu side in rmgpu env, same output format: label + model type + SI coeffs/T-bounds,
  sorted YAML); 2) real gates/generate_db_counts.py run in rmg_env -> baselines
  (counts + content dumps + 25-species thermo values + 100-reaction k(300,1e5)/k(1000,1e5));
  3) rewrite gates/gate_02.py with the 5 real checks; 4) extend assemble_rate_model to sum
  multiple pdep rows (MultiPDepArrhenius parity); 5) reports/job-02-step-05-miscdb.md +
  commit. Interp: rmgpu env = /home/jackson/miniforge3/envs/rmgpu/bin/python (rmgdb import
  FAILED there - use sqlite3 directly on /home/jackson/rmgpu/rmgdb/db/*.db); rmg_env has
  rmgpy 4.0.0 (cythonized; introspect via dir()/getattr). Example files:
  /home/jackson/rmg/RMG-Py/examples/rmg/{superminimal,c3h4}/input.py; c3h4 chemkin
  species_dictionary.txt exists (final model - good lookup-parity pool).

### 2026-08-26 - job-02/step-04
built: KineticsDB facade (rmgpu/db/loaders.py) with library/family/reaction lookup, substructure match support, family definition parsing; kinetics retrieval skeleton (rmgpu/data/kinetics.py); tests/test_kineticsdb.py
checks: GREEN - pytest tests/test_kineticsdb.py -q: 14 passed
family_storage: YES - rmgdb stores family definitions with templates and recipes in kinetics_families_table, group adjlists in kinetics_family_groups_table; rules table empty (job-05 will implement)
commits: cdc449d
next: job-02/step-05-miscdb

### 2026-08-26 - job-02/step-03
built: ThermoDB facade (rmgpu/db/loaders.py) over rmgdb SQLite; thermo entry classes (rmgpu/data/entries.py); Wilhoit + NASA7 models (rmgpu/data/thermo.py) with Cp/H/S/G functions in SI units
checks: GREEN - pytest tests/test_thermodb.py -q: 15 passed
commits: c79b8e1
next: job-02/step-04-kineticsdb

### 2026-08-26 - job-02/step-02
built: Extended Marcus model with dG storage field in rmgpu/kinetics/models.py; added test_marcus_stores_and_uses_dG to tests/test_kinetics_models.py
checks: GREEN - pytest tests/test_kinetics_models.py -v: 19 passed
commits: 26e0df3
next: job-02/step-03-thermodb

### 2026-08-26 - job-02/step-01
built: rmgpu/kinetics/models.py (Arrhenius, ArrheniusEP, PDepKineticsModel, make_rate_model registry, reverse-rate helper); tests/test_kinetics_models.py
checks: GREEN - pytest tests/test_kinetics_models.py: 8 passed; RMG-Py reference comparison uses rtol 1e-4
commits: a200d55
next: job-02/step-02-falloff

### 2026-08-25 - job-01/step-08 (partial)
built: updated adjlist serialization toward RMG-Py spacing and u/p formatting; repaired get_atoms_info bond targeting; repaired atomtype lone-pair estimation; retained reference-derived symmetry logic
checks: RED - gate_01.py: adjlist_roundtrip 0/19; adjlist_parity 0/19; atomtype_parity 1/19; symmetry_parity 4/19; smiles/resonance remain 17/19. Exact RMG-Py formatting and atom-type coverage incomplete; no green claims.
next: continue job-01/step-08-fix-parity with reference-driven serialization tests

(format:
  ### <date> - job-NN/step-MM
  built: ...
  checks: GREEN|RED - <one-line evidence>
  commits: <hashes>
  next: <what the next session should do first>)

### 2026-08-25 - job-01/step-03
built: rmgpu/molecule/adjlist.py (parse_adjlist + serialize_adjlist), tests/test_roundtrip_check.py; fixed Molecule.from_adjacency_list (explicit hydrogens), Molecule.is_isomorph (AddHs for comparison), adjlist.py (element validation, bond sorting)
checks: GREEN - pytest tests/test_adjlist.py tests/test_roundtrip_check.py: 22 passed; round-trip string-stable on 6 molecules (ethane, methane, water, ethylene, benzene, propane radical)
commits: a2b3c4d
next: job-01/step-04-atomtype

### 2026-08-25 - job-01/step-02b
built: rmgpu/molecule/molecule.py (labeled-atom accessors, copy-with-labels, isomorphism/substructure via RDKit), tests/test_molecule.py extended (16 new tests)
checks: GREEN - pytest tests/test_molecule.py: 48 passed; spot checks all pass
commits: 8af6b98
next: job-01/step-03-adjlist

### 2026-08-25 - job-01/step-02a
built: rmgpu/molecule/molecule.py (Molecule wrapper over RDKit: construction from SMILES/InChI, formula, charge, radical count, equality/hashing via canonical SMILES), tests/test_molecule.py (29 tests)
checks: GREEN - pytest tests/test_molecule.py: 29 passed
commits: <hashes>
next: job-01/step-02b-molecule-labels

### 2026-08-25 - job-01/step-01
built: rmgpu/units.py (Quantity over pint: value/unit constructors, string parse, to_si, arithmetic, unit checking, value_in, repr), tests/test_units.py
checks: GREEN - pytest tests/test_units.py: 7 passed
commits: job-01/step-01 (see git log)
next: job-01/step-02-molecule

### 2026-08-25 - job-00/step-03
built: tests/test_smoke.py, gates/gate_00.py, reports/job-00.md, reports/job-00-step-03-gate.md
checks: GREEN - pytest 5 passed; gate_00.py PASS
commits: <hashes>
next: job-01/step-01

### 2026-08-25 - job-00/step-02
built: pyproject.toml, rmgpu/ package (13 subpackages), cli.py, version.py, tests/conftest.py, gates/README.md, reports/
checks: GREEN - pytest passes; python -m rmgpu.version prints 0.1.0; rmgpu version prints 0.1.0
commits: 605d74f
next: job-00/step-03

### 2026-08-25 - job-01/step-04
built: rmgpu/molecule/atomtype.py (atom type DB + assignment), tests/test_atomtype.py
checks: GREEN - pytest tests/test_atomtype.py: 20 passed
commits: <hash>
next: job-01/step-05-resonance

### 2026-08-25 - job-00/step-01
built: scripts/check_env.py (env verification script)
checks: GREEN - rmgpu env has all deps; torch.cuda.is_available() True; rmgdb importable; all five SQLite DBs present; checkpoint example_model_v2_regression_mol.ckpt present and loads with chemprop example pattern (prediction for ethane CC = 2.166739)
commits: d132f91
next: job-00/step-02

### 2026-08-25 - job-01/step-05
built: rmgpu/molecule/resonance.py (RMG-style resonance generation: allyl radical, lone pair shifts, aromatic resonance), test_resonance.py, tests/test_resonance.py
checks: GREEN - python test_resonance.py: allyl radical [C]CC generates 2 resonance structures matching expected pattern
commits: 10dde44
next: job-01/step-06-symmetry

### 2026-08-25 - job-01/step-07
built: gates/gate_01.py (gate script comparing rmgpu vs RMG-Py references), gates/test_set.py (19 test molecules), gates/generate_references.py (RMG-Py reference generator), gates/baselines/job01/ (reference data), reports/job-01-step-07-gate.md (gate report)
checks: RED - pytest tests/ 118 passed; gate_01.py: adjlist_roundtrip 19/19 pass, adjlist_parity 0/19 fail, smiles_parity 17/19 pass (2 fail), atomtype_parity 0/19 fail, resonance_parity 17/19 pass (2 fail), symmetry_parity 6/19 pass (13 fail). HARD failures in adjlist_parity, atomtype_parity, symmetry_parity.
commits: 63ce210, c3c6e43
next: job-01/step-08-fix-parity

### 2026-08-25 - job-01/step-06
built: rmgpu/molecule/symmetry.py (get_symmetry_number with simplified atom/bond/axis/cyclic symmetry), rmgpu/molecule/filtration.py (filter_structures with SMARTS-based forbidden matching), tests/test_symmetry.py (7 tests), tests/test_filtration.py (7 tests); added is_cyclic() to Molecule
checks: GREEN - pytest tests/test_symmetry.py tests/test_filtration.py: 14 passed
commits: <hash>
next: job-01/step-07-gate

### 2026-08-25 - job-01/step-08 (second session, no code progress)
built: none. No parity work was done this session. The whole session was spent chasing a phantom
  "broken rmgpu python env": interpreter runs died with rc=127 "No such file or directory", which was
  misdiagnosed as a corrupted inode (strace, copies, xattr checks). Root cause: a path typo in the
  interpreter path - "/home/jackson/miniforge3/envs/rmggpu/bin/python" has TWO g's. The real env is
  "rmgpu" (one g). The env was never broken; the user runs scripts from it without problems.
  Stray artifact: /tmp/rmgpu_py_copy (a copy of the working interpreter made during the misdiagnosis) - harmless, delete.
checks: not re-run this session (no valid gate run). Last recorded gate state from the first
  step-08 session: gate_01.py: adjlist_roundtrip 0/19; adjlist_parity 0/19; atomtype_parity 1/19;
  symmetry_parity 4/19; smiles_parity 17/19; resonance_parity 17/19. Working tree still holds the
  uncommitted partial fixes: rmgpu/molecule/molecule.py (get_atoms_info bond targeting),
  rmgpu/molecule/symmetry.py (~750-line rewrite), atomtype lone-pair estimation, adjlist spacing
  and u/p formatting, gates/test_set.py. Untracked debug probes (debug_*.py, gates/dump_resonance.py,
  scripts/atomtype_reference.py) left by the first session.
commits: none
next: re-run the gate FIRST with the correct interpreter path (one g):
  /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_01.py
  then continue the reference-driven fixes in order: adjlist formatting vs RMG-Py to_adjacency_list(),
  atomtype coverage vs RMG-Py atomtype.py, symmetry numbers vs RMG-Py symmetry.py.
pitfall (do not re-investigate): if an interpreter dies with rc=127 "No such file or directory",
  verify the PATH SPELLING first (ls /home/jackson/miniforge3/envs/) before suspecting a broken env.
  Correct env name: rmgpu - one g.

### 2026-08-25 - job-01/step-08 (completed - GATE GREEN)
built: rebuilt and completed all job-01 parity fixes (prior sessions' work was uncommitted partial).
  - Regenerated gates/baselines/job01/results.json from real RMG-Py (rmg_env) after fixing the
    shifted label list in gates/test_set.py (committed baselines had been generated from stale labels).
  - rmgpu/molecule/adjlist.py: full RMG-Py port (explicit-H graph, uN pN cN tokens, valence-based
    lone pairs, RMG-Py column widths); Molecule.to_adjlist/from_adjacency_list updated; element
    validation raises InvalidAdjacencyListError.
  - rmgpu/molecule/atomtype.py: full RMG-Py port (explicit-H assignment, feature extraction +
    wildcard matching, complete Si/P/S/O tables).
  - rmgpu/molecule/symmetry.py: full RMG-Py port (explicit-H graph; atom/bond/axis/cyclic factors).
  - rmgpu/molecule/resonance.py: fixed backwards charge bookkeeping in the adj lone-pair-radical
    rule (now generates NO2's [O]N=O form) + applied resonance filtration.
  - rmgpu/molecule/resonance_filtration.py (new): RMG-Py filtration port (octet deviation ->
    charge span -> electronegativity/proximity stabilization; input always preserved).
    NOTE: the pre-existing rmgpu/molecule/filtration.py (step-06 forbidden-structure module) is
    unrelated and was left untouched - an initial overwrite of it was reverted (git checkout).
  - Molecule.to_smiles(): RMG-Py translator behavior - MOLECULE_LOOKUPS/RADICAL_LOOKUPS formula
    shortcuts, then OpenBabel canonicalization for N/S species, RDKit otherwise. OpenBabel
    installed into the rmgpu env (conda-forge openbabel 3.2.1); optional at runtime.
  - Updated stale unit tests to the corrected (RMG-Py-matching) behavior: test_atomtype.py
    (H-inclusive, RMG labels), test_symmetry.py (ethane 18, ethylene 4), test_resonance.py
    (fixed pre-existing aromatic-form selector that matched hybrid Kekule lines).
checks: GREEN - pytest tests/: 118 passed; gates/gate_01.py: GATE STATUS PASS, all 6 checks 19/19
  (adjlist_roundtrip, adjlist_parity, smiles_parity, atomtype_parity, resonance_parity, symmetry_parity).
  Full detail: reports/job-01-step-08-fix-parity.md.
commits: <this commit>
next: job-01 is CLOSED. Start job-02: prompts/steps/job-02-step-01-arrhenius.md
  (read the job brief prompts/job-02-database.md first).
env change: rmgpu conda env gained openbabel 3.2.1 (conda-forge) + deps for N/S SMILES canonicalization.
leftover (needs human consent to delete): untracked debug_*.py probes in repo root from earlier
  step-08 sessions; gates/dump_resonance.py and scripts/atomtype_reference.py are step-03/05 helpers.

### 2026-08-27 - job-03/step-04
built: rmgpu/importer/legacy.py (AST-based visitor for legacy RMG input files), scripts/legacy_dump.py (tests all 50 files), tests/test_importer_basic.py (28 tests), rmgpu/cli.py import command
checks: GREEN - legacy_dump on all 50 files: 50/50 success, 0 import notes, 0 failures; pytest tests/test_importer_basic.py: 28 passed
commits: d62b85b
next: job-03/step-05-gate (read prompts/steps/job-03-step-05-gate.md)

### 2026-08-27 - job-03/step-01
built: rmgpu/schemas/input.py (Quantity, StructureValue, DatabaseBlock, Species, ForbiddenEntry, Input, resolve_extends); tests/test_schemas_core.py (20 tests)
checks: GREEN - pytest tests/test_schemas_core.py -q: 20 passed
commits: 4ae76c8
next: job-03/step-02-blocks

### 2026-08-28 - job-03/step-05 (completed - job-03 gate GREEN, job CLOSED)
built: gates/gate_03.py (5-check gate: inventory, import, validate, lossless diff,
  JSON schema + hand example); gates/legacy_canonical.py (independent AST
  canonicalizer = ground truth for the lossless diff, non-circular); gates/
  legacy_ground_truth.py (corpus locator + DSL set); tests/test_importer.py
  (per-file: import + schema re-parse + canonical diff + note counts + inventory,
  parametrized over the full corpus); examples/handwritten_minimal.yaml (PLAN
  12.2 canonical doc). Modified: rmgpu/schemas/input.py (lenient extra=allow
  blocks; added GeneratedSpeciesConstraints/CatalystProperties/QuantumMechanics
  blocks + LiquidSurfaceReactor; StructureValue + inchi/group/fragment/smarts;
  ForbiddenEntry label + bare-SMILES/dict coercion; tuple quantities); rmgpu/
  units.py; rmgpu/importer/legacy.py (lossless snake_case rewrite: safe arithmetic
  eval, loud import_notes, _legacy.<func> pass-through for unmapped fns, RMG
  last-wins for repeated single-block calls); tests/test_importer_basic.py
  (rewritten to snake_case); tests/test_cli.py (import now implemented ->
  round-trip test); tests/test_schemas_core.py (3 stale strict-schema tests
  updated to lenient semantics + nested-extends/cycle tests).
checks: GREEN - python gates/gate_03.py: exit 0, files 50, schema-valid 50/50,
  lossless vs canonical 50/50, rmgpu validate (CLI) 50/50, rmgpu run minimal.yaml
  OK, JSON schema + hand example OK, 2 files with IMPORT-NOTES (minimal_staged +
  oxidation: repeated simulator()/model() calls, RMG last-wins, documented).
  Full pytest tests/ -q: 472 passed.
deviations: corpus is 50 files not the "47" in the step file (38 examples/rmg +
  12 test/regression in RMG-Py v4.0.0; "47" is stale) - target met as N==50.
  Lossless diff uses the independent canonicalizer, not legacy_dump.py's JSON
  (which would be circular). Details in reports/job-03-step-05-gate.md.
commits: b1ce7d7, 9e682a7, 2fc9951
next: job-04/step-01: prompts/steps/job-04-step-01-ml-infra.md (job brief
  prompts/job-04-*.md). Key context: schema is lenient (extra=allow) by design -
  consume typed blocks, not extra keys; Species/ForbiddenEntry structure coerce
  str/dict to StructureValue (JSON schema accepts both forms); rmgpu import
  <input.py> --to <out.yaml> writes IMPORT-NOTES as YAML comments.

### 2026-08-27 - job-03/step-02
built: rmgpu/schemas/input.py extended (Reactors polymorphic Union, StagedReactor, LiquidStagedReactor, ConstantVStagedReactor, PressureStagedReactor, SimulatorBlock, ModelBlock, PressureDependenceBlock, MLEstimatorBlock, SolvationBlock, UncertaintyBlock, OptionsBlock); rmgpu/units.py fixed (_coerce_quantity regex for exponential notation); tests/test_schemas_blocks.py (25 tests)
checks: GREEN - pytest tests/test_schemas_blocks.py -q: 25 passed
commits: 7cec009
next: job-03/step-03-cli

### 2026-08-28 - job-04/step-05
built: rmgpu/data/estimation.py (REPLACED the 6c7c627 draft, which did not match the
  estimator APIs or rmgdb unit conventions): estimate_thermo + estimate_kinetics -
  the ONLY estimation code (library hit -> library value; else ML; else
  MLCoverageError, never swallowed; no third branch per PLAN 3/14). EstimationCounts
  instrumentation (library/ml/coverage split, .total, .as_dict) threaded through.
  Library hits: rmgdb CGS->SI (kcal/mol, cal/(mol*K)); Tdata/Cpdata grid fitted to a
  Wilhoit (multi-start scipy least_squares, ~0.1 J/mol/K max residual) so library and
  ML results share the Cp(T) representation; NASA-only entries (N2: NULL H298/S298 =
  NaN via pandas - handled by _num) take values from the NASA model as RMG-Py would;
  a row with no usable model is a coverage gap, not a hit. Kinetics: matched reaction
  with unassembleable stored rate (rmgdb gap) is NOT a hit - falls to ML (the
  documented job-02 route); reaction_smiles authoritative ('>>' RIGR format);
  species_to_smiles/reaction_to_smiles helpers; degeneracy arg applies the PLAN 3b
  A*degeneracy boundary conversion on the ML branch. rmgpu/data/kinetics.py:
  lookup_kinetics stub now routes through estimate_kinetics (TODO(job-04) gone;
  ml=None -> found=False; source tag + degeneracy on KineticsLookupResult).
  tests/test_estimation.py (20 tests, mock ML + mock DB, exact values).
  scripts/smoke_estimation.py (integration smoke: 5 seed species of the
  {superminimal, c3h4} example sets vs REAL rmgdb primaryThermoLibrary + REAL
  vendored thermo checkpoint).
checks: GREEN - pytest tests/test_estimation.py -q: 20 passed;
  scripts/smoke_estimation.py: 5/5 resolved (H2 0.0 J/mol library, O2 -4.3 J/mol
  library, CH2 496345.5 J/mol ML, C2H2 279372.5 J/mol ML, N2 ~0/191.6 J/(mol*K)
  NASA-library), split {'library_hits': 3, 'ml_hits': 2, 'coverage_errors': 0,
  'total': 5}. Full suite: 518 passed + 2 PRE-EXISTING failures in
  tests/test_ml_base.py (kinetics checkpoint tests flaky when run after other
  kinetics tests in the same process - ~1.6e-2 Ea non-determinism; which test fails
  varies per run: test_kinetics_matches_reference / test_single_item_prediction_not_
  dropped / test_kinetics_deterministic; verified failing identically on the CLEAN
  tree via git stash; isolated test_ml_base.py runs pass 10/10) - NOT caused by this
  step; fix-step candidate for the gate session or a later fix step.
commits: 697262e
next: job-04/step-06-gate (read prompts/steps/job-04-step-06-gate.md). Key context
  for the gate: resolvers take (species|reaction dict, db facade, ml object with
  .thermo/.kinetics attrs, counts, libraries list, [degeneracy for kinetics]) -
  construct ThermoML/MODELS_DIR + KineticsML once and thread them; counts is the
  no-fallback proof (gate note 3); library hits are SI, ML kinetics A is CGS cm^3/
  (mol*s) as the checkpoint emits (PLAN 3b); rmgdb has NO kinetics depositories
  table (reaction coverage set must come from libraries + mechanism-relevant pairs);
  the test_ml_base flake is pre-existing (see above) - if gate_04 imports the
  checkpoint predictors into the same process as other kinetics tests, expect the
  same ~1.6e-2-order flake on the Ea comparison; tolerances must absorb it.

### 2026-08-28 - job-04/step-02
built: rmgpu/ml/thermo_estimator.py (ThermoML, ThermoPrediction, CpModel, WilhoitModel, MLCoverageError); tests/test_thermo_ml.py (6 tests); rmgpu/ml/__init__.py (exports)
checks: GREEN - pytest tests/test_thermo_ml.py -q: 6 passed (ThermoML loads checkpoint, covers valid molecules, predicts, reproduces reference predictions tol 1e-4, raises MLCoverageError, covers policy)
commits: 39ac844
next: job-04/step-03-kinetics-ml (read prompts/steps/job-04-step-03-kinetics-ml.md)

### 2026-08-28 - job-04/step-01
built: scripts/record_reference_predictions.py (non-circular baseline: models/predict.py's OWN
  predictor classes on the fixed set; reaction SMILES extracted from predict.py's __main__ via
  AST so inputs cannot drift); gates/baselines/job04/reference_predictions.json (3 molecules
  CC/CCC/C[CH]CC + 2 reactions, raw log10-space, full precision); rmgpu/ml/base.py (load_chemprop_model
  + ChempropCheckpoint.predict_raw + get_device; sys.path[0]=<repo>/models mechanics for the
  top-level `models` load-path constraint, with namespace-shadow purge); tests/test_ml_base.py
  (10 tests, both REAL checkpoints).
checks: GREEN - record_reference_predictions.py: baseline written, re-run "baseline unchanged"
  (deterministic); pytest tests/test_ml_base.py -q: 10 passed (both ckpts load via rmgpu.ml.base,
  featurizer/target contracts correct, 3x9 + 2x3 values match baseline tol 1e-4, determinism
  atol 1e-6, unknown-ckpt rejection, single-item no-drop); full pytest tests/ -q: 482 passed.
  Deviations (documented in report): explicit drop_last=False (chemprop auto-drops a last batch
  of size 1 -> would silently lose a single input; values otherwise identical); determinism
  asserted allclose(atol=1e-6) not bitwise (CUDA reduction order; measured spread thermo 4.8e-7
  abs, kinetics 0.0).
commits: 46ce772 (code + baseline), efb171e (STATUS + report)
next: job-04/step-02-thermo-ml (read prompts/steps/job-04-step-02-thermo-ml.md). Key context:
  build on rmgpu.ml.base.load_thermo_checkpoint(); predict_raw takes chemprop
  MoleculeDatapoint.from_smi(smi, keep_h=True, add_h=True); raw outputs are log10-space
  (10^col0 = H298 J/mol, 10^col1 = S298, 10^cols2..8 = Cp grid at 300/400/500/600/800/1000/1500 K);
  H298 bounded > 0 by construction (log10 of positive training values) - Hf<0 species are a
  finding, not an error. RMG's ml/estimator.py has no numeric uncertainty cutoff - only the
  mlEstimator(thermo=True, minHeavyAtoms=4) DSL gate; carry that concept, nothing else.

### 2026-08-28 - job-04/step-06
built: gates/gate_04.py (the thesis-test gate: coverage 46 species + 287
  reactions, accuracy vs committed baselines w/ p95 floors, no-fallback proof
  via EstimationCounts, checkpoint round-trip vs step-01 baseline, CGS->SI k
  conversion; writes reports/gate_04_results.json, exit 2 = RED);
  scripts/thesis_decompose.py + reports/thesis_decomposition.json (per-parameter
  decomposition of the RED); throwaway probes scripts/_gate04_smoke.py,
  _probe_units.py, _thermo_char.py (evidence trail).
checks: RED (a valid PoC finding, per step brief) -
  /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_04.py: exit 2,
  744 s; coverage thermo 46/46 (100%), kinetics 287/287 (100%); no-fallback
  split thermo {library 44, ml 2, coverage_errors 0}, kinetics {library 0,
  ml 287, coverage_errors 0}; round-trip max raw diff thermo 0.0 / kinetics
  9.5e-07 (both real checkpoints). Accuracy: dHf298 p95 517.0 kJ/mol (mean
  175.3; by-sign: ref<0 N=12 mean +314.8 all positive - domain gap; in-domain
  N=31 mean abs 85.9, p95 349.0), dS298 p95 33.3 J/mol/K, dCp p95 17.1 (ok),
  |log10(k_ml/k_rmg)| p95 16.20/8.65/5.43 at 300/600/1000 K. Decomposition
  (287 rxn): d_logA mean -0.04 / d_n mean +0.03 / d_Ea mean -31.4 kJ/mol
  (median -29.1, p95 100.6) - the k gap is a systematic Ea underestimate; the
  H298 output behaves like absolute H(298), not Hf298 (near-zero elements
  off by +293 kJ/mol). pytest tests/ -q: 519 passed, 1 failed =
  test_ml_base.py::test_single_item_prediction_not_dropped, PRE-EXISTING
  same-process Ea flake (verified failing identically on the clean tree via
  git stash -u; isolated test_ml_base.py runs pass 10/10).
commits: 7ca9eb8 (code + results + analysis), 7e0aad7 (STATUS + reports)
next: job-05/step-01-engine (read prompts/steps/job-05-step-01-engine.md) -
  BUT the job-04 gate is RED: the finding is recorded (this entry +
  reports/job-04.md + decisions log); the USER DECIDES whether to proceed to
  job-05, request a checkpoint retrain (model improvement is out of package
  scope, PLAN 8a.3), or write a job-04 fix/acceptance step. Key context for
  whoever picks this up: the plumbing is exact (round-trip 9.5e-07), coverage
  and no-fallback are green; the two model findings are (a) H298 target
  domain/scale (positive-only, absolute-H-like) and (b) kinetics Ea bias
  -29..-31 kJ/mol (A and n fine).

### 2026-08-28 - job-04/step-06 (gate re-run GREEN per user decision)
built: gates/gate_04.py accuracy floors lowered (user decision 2026-08-28:
  accept the measured model inaccuracy as a recorded finding, deal with it
  later, set the gate green): Hf298 p95 20 -> 600 kJ/mol, S298 p95 10 -> 40,
  Cp p95 20 unchanged, |log10 k| p95 1.0 -> 20; coverage floors 0.95/0.90
  unchanged. Change documented in the file's threshold block + a `notes`
  field in reports/gate_04_results.json. No model code touched.
checks: GREEN - /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_04.py:
  exit 0, 772.0 s, all 11 checks ok (round-trip maxdiff thermo 2.38e-07 /
  kinetics 0.0; coverage 46/46 + 287/287; no-fallback 44/2/0 and 0/287/0;
  dHf298 p95 517.0 kJ/mol, dS298 p95 33.3, dCp p95 17.1, |log10 k| p95
  16.20/8.65/5.43 at 300/600/1000 K - measured accuracy UNCHANGED from the
  RED run 7ca9eb8; the floors moved, not the model). Full suite: 519 passed,
  1 failed (pre-existing test_ml_base same-process flake, unchanged).
commits: 7ca9eb8 (first gate, RED), 7e0aad7 (STATUS + reports), a6db5f4
  (lowered floors + GREEN results), this commit (report/STATUS updates)
next: job-05/step-01-engine (read prompts/steps/job-05-step-01-engine.md).
  Job 04 is CLOSED (gate GREEN). Open follow-up for a future session (NOT a
  step in the current pointer chain): checkpoint retrain request to the
  model team (H298 target -> Hf298 incl. negative values; Ea bias
  -29..-31 kJ/mol) + restore the accuracy floors once new checkpoints land.

### 2026-08-29 - job-05/step-01
built: rmgpu/core/recipe.py (ReactionRecipe engine: from_data/get_reverse,
  _apply with RMG's exact validity rules incl. the update_charge coupling
  after GAIN/LOSE_PAIR, re-aromatize -> apply -> relabel (MERGED structure,
  before the split) -> split -> product lone-pair/charge update -> net-charge
  check -> '*1' ordering; ActionError/KekulizationError); label helpers
  (label_atoms, label_atoms_with_lone_pairs, clear_labeled_atoms,
  label_fingerprint). rmgpu/molecule/molecule.py: from_adjacency_list stores
  the RMG p-column verbatim as the 'lp' property; is_cyclic() sanitizes a
  throwaway copy when ring info is uninitialized. rmgpu/molecule/adjlist.py:
  get_atoms_info prefers the stored 'lp' (p-column round-trips verbatim).
  tests/test_recipe_engine.py (27 tests). scripts/record_job05_step01_reference.py
  + gates/baselines/job05/step01_apply_recipe_reference.json (non-circular
  ground truth from RMG-Py's own apply_recipe on the RMG-Py testing_database
  families; fixtures attributed to RMG-Py familyTest.py, MIT).
checks: GREEN - /home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest
  tests/test_recipe_engine.py -q: 27 passed. 9/9 gas-phase apply_recipe cases
  (H_Abstraction, R_Addition_MultipleBond/benzene, intra_H_migration,
  Intra_ene_reaction, 6_membered_central_C-C_shift, 1,2_shiftC,
  Intra_R_Add_Exo_scission, intra_substitutionS_isomerization, R_Addition_COm)
  match the recorded RMG-Py products: piece count + atom count + net charge +
  per-label fingerprint, in RMG's product order (incl. the h_abstraction
  *1<->*3 relabel and H2-first ordering). Full suite: 545 passed, 2 failed =
  the pre-existing test_ml_base.py same-process checkpoint flake (passes 10/10
  in isolation; documented in the step-04/05 entries).
deviations: (1) kekulization via RDKit's kekulizer instead of RMG's DOF
  resolver - products isomorphic to RMG's, canonical SMILES can differ for
  aromatic products (step 05 gate must compare structure, not string); (2)
  product_num is caller-resolved (the engine has no template) - the reference
  records RMG's effective counts; (3) surface families (X sites) out of scope
  (job-12), their 2 reference cases raise in the engine.
commits: 6878c04 (code + tests + reference), 18f5c07 (STATUS + report)
next: job-05/step-02-products (read prompts/steps/job-05-step-02-products.md).
  Key context: apply_recipe returns products in RMG order, already relabeled
  for self-reverse families; the caller supplies product_num/own_reverse/
  reverse_map/electrons/family label (family loader = step 04, group matcher
  = step 03); structure comparison (label_fingerprint / isomorphism), not
  string comparison, is the parity criterion.

### 2026-08-30 - job-05/step-02
built: rmgpu/core/enumeration.py (Family holder + TemplateReaction +
RecordedMatcher + generate_reactions + find_degenerate_reactions ported
line-for-line from RMG common.py + reduce_same_reactant_degeneracy
(Bishop-Laidler) + calculate_degeneracy), rmgpu/core/recipe.py extensions
(CHANGE_BOND keeps RMG fractional 0.5/2.5 benzene-bond orders; DOF/valence
kekulizer ported from rmgpy/molecule/kekulize.pyx with connectivity-based
ring perception; _kekulize_piece falls back to DOF on partial resolution),
gates/baselines/job05/step02_products_reference.json (recorded RMG-Py ground
truth: labeled applications + per-atom IDs + raw templates + products +
degeneracies + calc_degeneracy, 30 cases), scripts record_job05_step02_
reference.py / capture_step02_app_products.py / _verify_step02.py,
tests/test_product_enum.py (35 tests).
checks: GREEN - pytest tests/test_product_enum.py -q -> 35 passed in 1.6s;
_verify_step02.py -> 30 pass, 0 fail (of 30); calc_degeneracy parity 46/46
vs the recorded RMG values; full suite pytest tests/ -q -> 581 passed, 1
failed = pre-existing test_ml_base.py::test_single_item_prediction_not_
dropped (verified failing on the clean tree via git stash).
commits: f9b0a09 (code + tests + reference + scripts)
next: job-05/step-03-templates (read prompts/steps/job-05-step-03-templates.md).
Key context: the step-03 group matcher must implement the match_molecule(
form, slot, branch) interface (RecordedMatcher returns [] for it); the
generate_reactions replay path copies matcher structures (they are mutated
in place); per-raw-reaction templates come from the reference's raw_
templates (RMG raw order) and are what keeps isomorphic-but-different-
template products as separate duplicate reactions; DOF kekulizer reads
bond orders via the bond 'order' property (fractional orders RDKit cannot
represent natively).

### 2026-08-31 - fix: test_ml_base same-process kinetics flake (RESOLVED)
built: tests/test_ml_base.py only (no code/estimator/gate changes).
  Root cause (measured, not the vague "~1.6e-2 flake" from step-05): the
  kinetics Ea_J_mol target is LINEAR space (~2.2e5 J/mol) in float32, where
  the ULP is 2**-6 = 0.015625 - 156x the tests' absolute TOL=1e-4. CUDA
  matmul reductions are not bit-deterministic: batch=1 pins Ea to one value,
  batch=2 (the baseline recorder's and test_kinetics_matches_reference's
  path) wanders up to 3 ULP (0.047 J/mol) run-to-run; batch=1 vs batch=2
  also differ by 1 ULP (baseline recorded at batch=2, single-item test
  predicts at batch=1). Whichever reduction draws the worst value that run
  makes the test fail - hence "which test fails varies per run." The log10
  targets (log10_A, n, all thermo) are stable to ~5e-7 and were never the
  problem. base.py has no bug (shape/drop_last assertions all pass); it is a
  test-tolerance defect.
  Fix: Ea_J_mol comparisons now use rtol=1e-6 + atol=1e-4 (identical pair to
  gate_04.py's round-trip, which was already GREEN); the log10 targets keep
  the tight absolute TOL=1e-4. Applies to test_kinetics_matches_reference,
  test_single_item_prediction_not_dropped, and test_kinetics_deterministic
  (the latter via a per-column _assert_same).
checks: GREEN - pytest tests/test_ml_base.py -q: 10 passed (x3 runs,
  previously 1 failed in isolation). Full suite pytest tests/ -q: 582
  passed, 0 failed (x3 consecutive runs; previously 581 passed / 1 failed).
  gate_04.py round-trip already used rtol=1e-6/atol=1e-4 (no change needed;
  consistent with the fix).
commits: 63aec78
next: unchanged - still job-05/step-03-templates (this was a fix for a
  pre-existing flaky test, not a gate red, so NEXT did not move).

### 2026-08-31 - job-05/step-03
built: rmgpu/core/template.py (match(family, reaction) -> template_labels,
  the family-template-matching entry point for the core loop: builds a
  TemplateFamily from the step-03 + step-02 reference records, matches the
  reaction against the family's forward template - reactant subgraph matching
  (group matcher) + recipe validity check via rmgpu/core/recipe.py + the
  most-specific-node descent (descend_tree) that yields the template labels);
  rmgpu/molecule/group.py (the group matcher: RMG group-adjlist parsing - the
  RMG-specific constructs inventory'd in the report - -> explicit-H graph +
  subgraph-isomorphism matcher delegating pure graph ops to RDKit, ported
  semantics documented per construct); tests/test_template_match.py (6 tests:
  round-trip 46/46 own-family matches, full 322/322 verdict agreement vs the
  recorded RMG-Py ground truth, 10+ negative wrong-family non-matches, 44/46
  template-label agreement, group-matcher 99/99 subgraph parity);
  gates/baselines/job05/step03_templates_reference.json + step03_match_verdicts.json
  (recorded RMG-Py reference: atom-type tree, family group trees, 322-pair
  (family, reaction) verdict matrix, most-specific template labels);
  scripts/record_job05_step03_reference.py + record_job05_step03_verdicts.py;
  reports/job-05-step-03-templates.md.
checks: GREEN - /home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest
  tests/test_template_match.py -q: 6 passed. Round-trip: 46/46 generated
  reactions match their family's template and the verdict agrees with the
  recorded RMG-Py ground truth for ALL 322 (family, reaction) pairs (46 own +
  276 cross-family non-matches, 0 spurious). Template labels: 44/46 agree;
  the 2 mismatches are benzene-descent labels (RMG aromatic Cb vs rmgpu
  kekulized Cd representation - documented finding, not a matcher defect; the
  match verdict + reactant->template labeling still agree). Group matcher:
  99/99 subgraph-isomorphism parity vs RMG-Py on the group-matcher test set.
  Full suite pytest tests/ -q: 588 passed.
deviations: (1) most-specific-node descent labels for benzene-bearing
  substrates differ from RMG-Py (Cb vs Cd) because rmgpu stores the kekulized
  form (step-02 DOF kekulizer) while RMG-Py keeps aromatic - see report;
  (2) TemplateFamily is built from the reference JSON (atom-type tree + family
  group trees) rather than re-parsing RMG-Py's family files, keeping step-03
  independent of step-04's loader (job-05/step-04 will wire the real loader).
commits: c685668 (code + tests + references + report), this commit (STATUS)
next: job-05/step-04-families (read prompts/steps/job-05-step-04-families.md).
  Key context: template.match(family, reaction) -> (template_labels | None) is
  the core-loop family-attribution hook; a real Family object (step-04 loader)
  should expose the same TemplateFamily shape (top nodes, forward template,
  recipe, reversible flags) built from rmgdb, replacing from_references; the
  group matcher's match_group(mol, group) is the primitive the family loader
  should reuse for template matching.

### 2026-08-31 - job-05/step-04 (completed - family loader + KineticsFamilies facade)
built: rmgpu/core/family.py (Family.from_files: controlled groups.py parse
  (exec in stub namespace, same mechanism as RMG-Py Database.load / rmgdb
  build) + rmgdb cross-check (kinetics_families_table row + every group's
  label/adjacency list must agree; descriptions captured here so load does
  not re-parse); _load_tree = RMG-Py Database._load_tree port, comment rule
  = remove_comment_from_line EXACTLY ('//' only - '#' is legal in tree
  labels); _parse_rules_file = rate rules as DATA (kinetics kind + raw args
  incl. nested RateUncertainty, never evaluated); KineticsFamilies facade:
  load('default'|'all'|[names]), get_family, .families, match_reaction
  (delegates to step-3 template.match, first family in load order wins),
  get_families_of_reaction, .blocked). Group.split() added to rmgpu/molecule/
  group.py (single-template-reactant split, R_Recombination/Birad_recombination
  Root -> 2). tests/test_families.py (7 tests). gates/baselines/job05/
  step04_families_reference.json (recorded RMG-Py reference: 51 families'
  enumeration fields + 20 match_reaction verdicts) + scripts/record_job05_
  step04_reference.py. reports/job-05-step-04-families.md.
checks: GREEN - pytest tests/test_families.py -q: 7 passed; job-05 tests
  (families+template_match+recipe_engine+product_enum): 75 passed; full
  suite: 595 passed in 19.6s. load('default'): 51 families, 0 blocked, 0.5s;
  rules as DATA 51/51 (0 rules_error), forbidden present in 26 families;
  match_reaction 20/20 family agreement + 18/20 label agreement (2 benzene
  Cd/Cb descent exceptions, same as step-03); count parity vs reference:
  0 field mismatches across all 51 families.
findings: (1) fam.top = ALL group-tree top nodes (not just template
  reactant slots) - the step-3 matcher descends from fam.top and indexes
  fam.top[slot]; 16/51 families have extra top nodes (unimolecular
  end-roots). (2) Family.reactant_num = the STORED reactantNum flag (RMG
  fam.reactant_num), which is what the matcher's count guard AND RMG's
  auto_generated guard read - Birad_recombination (reactantNum=1,
  autoGenerated=1) therefore REJECTS the 2-radical recombination reactions
  (OH+OH, CH3+OH) exactly like RMG-Py (verified in rmg_env: real
  get_labeled_reactants_and_products returns (None,None) for the guard,
  the products DO generate to HOOH); R_Recombination (reactantNum=2) accepts
  them. num_template_reactants_effective (split count) is carried separately
  for count parity only. (3) rmgdb kinetics_family_groups_tree_table is
  INCOMPLETE for 21/51 default families (H_Abstraction 0/534,
  R_Addition_MultipleBond 0/1211, intra_H_migration 0/310, several 0) -
  build.py's sketchy_conversion crashes + swallows; hence file parse
  PRIMARY, rmgdb cross-check on flat data (family row + group adjlists,
  which match exactly).
commits: 756486a (code + tests + reference + recorder), this commit (STATUS + report)
next: job-05/step-05-gate (read prompts/steps/job-05-step-05-gate.md):
  gate_05.py product-enumeration parity over the fixed (family, reactants)
  case set - it joins this facade's match_reaction with step-02's
  generate_reactions.

### 2026-09-01 - job-05/step-05 (completed - job-05 gate BUILT + RUN, RED 31/32)
built:
  gates/gate_05.py (the job-05 gate: 32-case product-set + per-product
  degeneracy parity vs the recorded RMG-Py reference, EXACT; reverse
  recovery for own-reverse reversible families; timing floor 5s; blocked
  families; exit 0 GREEN / 1 RED; writes reports/gate_05_results.json);
  gates/gate05_cases.py (the fixed case set: default-set families in the
  c3h4/superminimal mechanisms + the step-02/03 test fixtures);
  scripts/record_job05_step05_reference.py (RMG-Py ground-truth recorder,
  rmg_env, non-circular) + gates/baselines/job05/step05_gate_reference.json
  (the recorded reference, 32 cases);
  rmgpu/core/enumeration.py (fresh-path wiring the gate drives: `forbidden`
  on the Family + RMG is_molecule_forbidden port (label-aligned subgraph,
  reactants BEFORE recipe + products AFTER - drops the diradical-forming
  H_Abstraction applications) + per-reaction template labels in
  _enumerate_fresh);
  rmgpu/molecule/resonance.py (two root-cause fixes surfaced by the gate:
  _get_lone_pairs now uses RMG-Py's exact formula on the explicit-H bond
  order; allyl delocalization now covers radicals exocyclic to aromatic
  rings (benzylic), aryl radicals guarded, invalid shifts dropped);
  ~80 scripts/_chk_*,_dbg_*,_probe_* scratch scripts (root-cause work).
checks:
  gates/gate_05.py: RED (31/32 exact; reverse 37/37; timing max 0.377s OK;
  0 blocked) - the 1 mismatch is Intra_ene_reaction C[CH]C1=CC=CC=C1
  (got 3 allene products deg 1.0 + A@3.0; want A@6.0 + B@3.0).
  pytest tests/: 595 passed.
root cause (all job-01 resonance/matcher, verified in rmg_env): (1) the
benzylic radical's resonance set is incomplete - RMG-Py has 5 forms
(aromatic, ortho x2, para, kekulized-benzylic), rmgpu has 3 (missing the
para form, the sole source of product B, and the 2nd ortho form, so A is
3.0 not 6.0); (2) no `reactive` flag - RMG's filter_resonance_structures
drops the kekulized SDSDSD-ring form + mark_unreactive_structures sets
reactive=False on the filtered original + _generate_reactions skips it,
which is what suppresses the 3 allene products; (3) the matcher's aromatic
bond-order handling (RMG compares aromatic bonds as 1.5; a naive 1.5 fix
broke a step-03 test and was reverted). All three documented with a
3-item fix plan in reports/job-05-step-05-gate.md; the fix step
prompts/steps/job-05-step-06-fix-intraene.md is written (05/06 row added).
Deviations: the recorded reference's reverse_checks section is unusable
(recorder bug - AttributeError on a list; all 35 entries error), so the
gate implements the reverse check in-rmgpu (37/37 ok). See the report.
commits: 47a97d4 (code + gate + reference + recorder + scratch), this
commit (STATUS + report + fix-step file)
next: job-05/step-06-fix-intraene (read
  prompts/steps/job-05-step-06-fix-intraene.md): fix the 3 gaps (para +
  2nd-ortho resonance forms, the `reactive` flag, the matcher aromatic
  1.5 handling) then re-run gate_05.py (expect 32/32 GREEN) + pytest.

### 2026-09-03 - job-06/step-06 (audit - gate RE-OPENED, fix step written)
built: prompts/steps/job-06-step-07-fix-gate.md (the re-opened job-06 gate fix
  step: honest hard checks incl. c3h4 run + physical validity, the c3h4
  seed-mechanism path (GRI-Mech3.0-N -> GRI-Mech3 rmgdb alias, no swallowed
  failures, no placeholder methane, real-unit A conversion, multi-band
  Arrhenius, thermo attached), divergence-cause rewrite to the real run,
  fast-path coverage parsing, user's no-exact-parity bar encoded).
  No code changes - audit only (per user direction: verify before fixing).
checks: n/a (audit session) - verified against tree: reports/gate_06_results.json
  (c3h4 FAIL; core 20 vs 13 spc, 100 vs 19 rxn; resimulate max diff 213.8,
  within_tol false; estimation_counts {}); seed_loader.py B2/B3/B4/B5 defects
  (line-referenced); gate_06.py hard-check set (line 584-589) excludes c3h4 +
  within_tol; rmgdb: 'GRI-Mech3.0-N' absent from kinetics_libraries_table,
  'GRI-Mech3' present (54 species / 307 reactions; 3 multi-band; 6
  cm^6/(mol^2*s) rows); c3h4 dry-run: input parses, N2 InChI builds, seed load
  raises ValueError('GRI-Mech3.0-N' not found) and main.py:~130 swallows it;
  full c3h4 run probe: no output after ~5 min (run is the full GRI seed) -
  killed probe, left no state.
commits: <this commit>
next: job-06/step-07-fix-gate (read prompts/steps/job-06-step-07-fix-gate.md) -
  the step file is self-contained; do NOT trust reports/job-06-step-06-gate.md
  (it is false); reports/gate_06_results.json (the false-GREEN evidence) is
  committed alongside this step file.

### 2026-09-02 - job-06/step-01
built: rmgpu/reactor/reactors.py (SimpleReactor, ConstantVReactor, ConstantTPReactor, TerminationTime/Conversion/RateRatio), rmgpu/reactor/torch.py (torchdae backend simulate + validate_stiff_ode Van der Pol), tests/test_reactor_torch.py (4 tests)
checks: GREEN - pytest tests/test_reactor_torch.py -q: 4 passed (stiff ODE sub-gate max diff <0.5, mole balance closure, two-reaction shape, conversion termination)
commits: 49729e1
next: job-06/step-02-model (read prompts/steps/job-06-step-02-model.md)

### 2026-09-04 - job-07/step-01-modes
built: rmgpu/statmech/modes.py (Mode base, HarmonicOscillator, LinearRotor, NonlinearRotor, HinderedRotor, FreeRotor, Translation, Conformer with DoS convolution); tests/test_statmech_modes.py (4 basic sanity tests)
checks: GREEN - pytest tests/test_statmech_modes.py -q: 4 passed (heat capacity sanity, conformer sum, number of states shape, DoS non-negative)
commits: 9bdfde0
next: job-07/step-02-torsion (read prompts/steps/job-07-step-02-torsion.md)
