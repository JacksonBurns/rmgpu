# job-04/step-05 report: estimation.py - library -> ML -> coverage error

Date: 2026-08-28
Commits: 697262e (replaces the draft committed as 6c7c627 by an earlier
interrupted session of this same step)

## What was built

- `rmgpu/data/estimation.py` (rewritten, ~530 lines) - the ONLY estimation
  code in the package (PLAN.md 3/14). `estimate_thermo(species, thermo_db,
  ml, counts, libraries)`: library hit -> SI library value; else ML
  (`ml.thermo`); else `MLCoverageError` (never swallowed, counter bumped).
  `estimate_kinetics(reaction, kinetics_db, ml, counts, libraries,
  degeneracy)`: library hit -> assembled rate model; else ML
  (`ml.kinetics`); else `MLCoverageError`. `EstimationCounts`
  (library_hits/ml_hits/coverage_errors/total/as_dict) is threaded through
  both - the thesis test's no-fallback proof (gate note 3).
- Library-hit plumbing in the same file: rmgdb CGS -> SI conversion
  (kcal/mol, cal/(mol*K), kJ/mol, dimensionless, R*T); Tdata/Cpdata grid
  fitted to a Wilhoit (multi-start scipy least_squares; measured max
  residual ~0.1 J/(mol*K) on the real primaryThermoLibrary grids) so
  library and ML results share the same Cp(T) representation; NASA-only
  entries take H298/S298 from the NASA model (N2 case: rmgdb leaves
  H298/S298 NULL -> NaN via pandas; handled); a row with no usable model
  is a coverage gap, not a hit.
- Reaction SMILES helpers: `species_to_smiles` (smiles or adjlist) and
  `reaction_to_smiles` (explicit `reaction_smiles` authoritative; else
  unmapped `reactants >> products`; the `>>` RIGR format the kinetics
  checkpoint expects; atom-mapping arrives with the recipe engine, job-05
  per PLAN 3b). `degeneracy` arg applies the PLAN 3b A = 10^pred *
  degeneracy boundary conversion on the ML branch.
- `rmgpu/data/kinetics.py` - the job-02 `lookup_kinetics` stub now routes
  through `estimate_kinetics` (the `TODO(job-04)` is gone). `ml=None`
  degrades to `found=False`; `KineticsLookupResult` gained a `degeneracy`
  field and the source tag.
- `tests/test_estimation.py` (20 tests, mock ML + mock DB, exact values):
  library-wins (ML never called), ML-when-no-library, MLCoverageError not
  swallowed (covers False, no ML configured, no structure), counts reflect
  the split exactly, SI conversion + Cp grid in J, NASA-only entry path,
  gap reaction (stored rate None) falls to ML, degeneracy conversion,
  `reaction_to_smiles` authority, and the `lookup_kinetics` adapter.
- `scripts/smoke_estimation.py` - the required integration smoke: the 5
  seed species of the {superminimal, c3h4} example sets (H2, O2 from
  examples/rmg/superminimal; CH2, C2H2, N2 from examples/rmg/c3h4) against
  the REAL rmgdb primaryThermoLibrary + the REAL vendored thermo
  checkpoint (models/chemeleon_thermo_662946.ckpt).

## Checks (commands + real results)

1. `pytest tests/test_estimation.py -q` -> **20 passed** in 6.9 s.
2. `/home/jackson/miniforge3/envs/rmgpu/bin/python scripts/
   smoke_estimation.py` -> **5/5 resolved**, split
   `{'library_hits': 3, 'ml_hits': 2, 'coverage_errors': 0, 'total': 5}`.
   Per species (Hf298 J/mol, S298 J/(mol*K), Cp300 J/(mol*K)): H2
   0.0 / 130.68 / 28.85 (library), O2 -4.3 / 205.12 / 29.39 (library),
   CH2 496345.5 / 169.03 / 27.84 (ML), C2H2 279372.5 / 235.17 / 53.27 (ML),
   N2 -0.0 / 191.61 / 29.13 (library, NASA-only entry). H2/O2 match their
   rmgdb stored kcal values (0.0, -0.0010244 kcal/mol) after conversion;
   N2's S298 = 191.6 J/(mol*K) matches the literature value for N2.
3. `pytest tests/ -q` (full suite) -> **518 passed, 2 failed** - both
   failures in `tests/test_ml_base.py` and PRE-EXISTING (see Findings).

## Findings

- **Pre-existing flake in tests/test_ml_base.py (kinetics checkpoint),
  NOT caused by this step.** Full-suite runs fail one of
  `test_kinetics_matches_reference` / `test_single_item_prediction_not_dropped`
  / `test_kinetics_deterministic` (varies per run) with an Ea discrepancy of
  ~0.0156 J/mol against a 1e-6 tol. Verified by `git stash -u` (clean tree
  6c7c627): the same 2 tests fail on the pristine tree in the full-suite
  order, while isolated `pytest tests/test_ml_base.py` passes 10/10.
  Mechanism: the kinetics checkpoint's same-process results drift when the
  estimator has already run other kinetics datapoints in the process
  (test order-dependent; different seed/accumulation state). This is a
  step-01/03-area issue (checkpoint determinism vs the baseline), out of
  step-05 scope; the gate session should either absorb ~1.6e-2 relative
  tolerance on Ea where it compares checkpoint outputs, or a fix step
  should address it.
- The earlier draft of this step (commit 6c7c627, "Update thermo
  estimation files") was replaced in place: it referenced nonexistent
  `KineticsPrediction` fields (`Ea_J_mol`, `Tmin`, `Tmax`, `source`,
  `degeneracy`), built `reactants -> products` SMILES (the checkpoint
  requires `>>`), skipped the CGS->SI conversion on library hits, did not
  scope library lookups to the run's libraries, and missed the
  `get_rate_model -> None` gap path.

## Reference reads beyond the step's list

- `rmgpu/ml/thermo_estimator.py`, `rmgpu/ml/kinetics_estimator.py`,
  `rmgpu/ml/base.py` (the step-02/03/01 outputs - required to match the
  real API; the step file lists "rmgpu/ml/ (steps 1-3)").
- `rmgpu/db/loaders.py` (ThermoDB/KineticsDB facades - listed).
- `rmgpu/data/entries.py`, `rmgpu/data/thermo.py` (entry classes, NASA/
  Wilhoit - direct dependencies).
- `models/predict.py` (reaction SMILES examples - direct dependency for
  the `>>` format).
- RMG-Py reference, minimal: `rmgpy/thermo/nasa.pyx` (NASA H(T)/S(T)
  formation-reference convention, for the NASA-only entry path).
- `rmgdb` SQLite: schema spot-checks (thermo.db views, kinetics.db
  tables - confirmed there is NO kinetics depositories table).

## Deviations from the step file

- "estimate_thermo(species, databases, ml)" - the signature takes the
  `thermo_db` / `kinetics_db` sub-facade rather than the `Databases`
  aggregate (the step's own reference list points at the job-02 facades,
  and the mock-DB tests need the sub-facade shape; the aggregate exposes
  them as `.thermo`/`.kinetics`, so job-06 threads `databases.thermo`).
- The integration smoke uses the 5 SEED species of the {superminimal,
  c3h4} example sets (H2, O2, CH2, C2H2, N2) - "5 superminimal species"
  is not a set that exists in the examples (superminimal has H2/O2 only,
  c3h4 adds CH2/C2H2/N2); this is the minimal example seed set and the
  one the gate will use for the mechanism-relevant reaction set.
- Everything else per the file.

## What the next step (step-06, the gate) should know first

1. Resolver signatures (the gate consumes these):
   `estimate_thermo(species, db.thermo, ml, counts, libraries)` ->
   `ThermoPrediction(Hf298 J/mol, S298 J/mol/K, Cp_model.get_heat_capacity(T),
   uncertainties{source})`;
   `estimate_kinetics(reaction, db.kinetics, ml, counts, libraries,
   degeneracy)` -> `(rate_model, degeneracy)`. `ml` is any object with
   `.thermo` / `.kinetics` attributes (None = not configured). `counts`
   is the no-fallback proof (gate note 3): `counts.as_dict()`.
2. Library hits are SI; ML kinetics A is CGS cm^3/(mol*s) exactly as the
   checkpoint emits (PLAN 3b boundary; the estimator applies A*degeneracy).
   The gate must be consistent about which side of that boundary it
   compares on.
3. rmgdb has NO kinetics depositories table (kinetics.db tables checked) -
   the gate's reaction-coverage set must come from the kinetics libraries
   (`kinetics_library_reactions_table`) + the mechanism-relevant reaction
   pairs, not a depository.
4. The pre-existing test_ml_base kinetics flake (~1.6e-2 relative Ea
   drift, order-dependent) will also bite any gate code that runs the
   kinetics checkpoint on several datapoints in one process - use
   tolerances that absorb it, or compare against the committed baseline in
   a fresh process.
5. The `libraries` argument is how the gate scopes lookups to the YAML
   `database:` block (primaryThermoLibrary for the examples).
