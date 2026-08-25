# STATUS - rmgpu work tracker

Single source of truth for work state across sessions. Every session updates this
file before committing. Do not delete entries; append and annotate.

## NEXT (the pointer - the human reads this first)

NEXT: prompts/steps/job-01-step-02b-molecule-labels.md
(When a step finishes, the session updates this pointer to the following
step's file, or to a small fix-step file written for a red gate. One step
at a time.)

## Job table (a job is done only when its GATE step is GREEN)

| Job | Title | Gate | Status |
|-----|-------|------|--------|
| 00 | Env + package skeleton + test scaffolding | smoke test (gate_00.py) | pending |
| 01 | Units + molecule layer | adjlist/atomtype/resonance parity (gate_01.py) | pending |
| 02 | Database layer via rmgdb + round-trip | entry-count + table hash vs RMG-Py (gate_02.py) | pending |
| 03 | YAML input schema + CLI + legacy importer | 47 example input.py -> yaml, lossless (gate_03.py) | pending |
| 04 | ML estimators + rate registry | thesis test: Hf298/S298/Cp, HPL k(T) vs RMG-Py (gate_04.py) | pending |
| 05 | Reaction recipe DSL + product enumeration | product sets + degeneracy parity (gate_05.py) | pending |
| 06 | Core/edge loop + torchdae reactor | superminimal + c3h4 core/edge vs RMG-Py (gate_06.py) | pending |
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
| 01/02b | Molecule wrapper: labels and structure queries | test_molecule.py (labels, isomorphism, substructure) | pending |
| 01/03 | Adjacency-list parser/serializer | test_adjlist.py | pending |
| 01/04 | Atom-type DB + assignment | test_atomtype.py | pending |
| 01/05 | Resonance structure generation | test_resonance.py | pending |
| 01/06 | Symmetry + filtration | test_symmetry/test_filtration | pending |
| 01/07 | Job-01 gate (round-trips vs RMG-Py) | gate_01.py PASS | pending |
| 02/01 | Rate models: Arrhenius family + registry base | test_kinetics_models.py | pending |
| 02/02 | Rate models: falloff, Chebyshev, Marcus, tunneling | test_kinetics_models.py | pending |
| 02/03 | ThermoDB facade + thermo models (Wilhoit/NASA7) | test_thermodb.py | pending |
| 02/04 | KineticsDB facade + family-definition storage | test_kineticsdb.py + storage finding | pending |
| 02/05 | Transport/StatMech/Solvation facades + job-02 gate | gate_02.py | pending |
| 03/01 | Input schema: core blocks | test_schemas_core.py | pending |
| 03/02 | Input schema: reactors + remaining blocks + extends | test_schemas_blocks.py | pending |
| 03/03 | CLI: run/validate/schema/version | test_cli.py | pending |
| 03/04 | Legacy importer: inventory + ast visitor | legacy_dump on 47 + visitor tests | pending |
| 03/05 | Job-03 gate (lossless import of 47 examples) | gate_03.py (target 47/47) | pending |
| 04/01 | ML infra: checkpoint inventory + synthetic test model | test_ml_base.py + synthetic ckpts | pending |
| 04/02 | ThermoML estimator (replacement of RMG's) | test_thermo_ml.py | pending |
| 04/03 | KineticsML estimator (Chemprop reactions) | test_kinetics_ml.py | pending |
| 04/04 | Rate registry: tunneling + forward/reverse wiring | test_kinetics_registry.py | pending |
| 04/05 | estimation.py: library -> ML -> coverage error | test_estimation.py | pending |
| 04/06 | Thesis test: coverage + accuracy + no-fallback proof | gate_04.py (numbers reported) | pending |
| 05/01 | ReactionRecipe engine (apply_recipe + labels) | test_recipe_engine.py | pending |
| 05/02 | Product enumeration (generate_reactions) | test_product_enum.py | pending |
| 05/03 | Template matching + group matcher | test_template_match.py | pending |
| 05/04 | Family loader + KineticsFamilies facade | test_families.py | pending |
| 05/05 | Job-05 gate (product enumeration parity) | gate_05.py (sets + degeneracy parity) | pending |
| 06/01 | Reactor definitions + termination + torchdae backend | test_reactor_torch.py + stiff sub-gate | pending |
| 06/02 | CoreEdgeReactionModel (enlarge/prune/screen) | test_core_model.py | pending |
| 06/03 | main.py: the job driver + the iteration loop | rmgpu run completes, deterministic | pending |
| 06/04 | Mechanism artifact schema + the output tree writer | test_output.py | pending |
| 06/05 | Chemkin writer + species dictionary | test_chemkin.py | pending |
| 06/06 | Job-06 gate (first real mechanism generation) | gate_06.py (superminimal + c3h4) | pending |
| 07/01 | Statmech modes: conformer, vibration, rotation | test_statmech_modes.py | pending |
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

## Session log (append newest at bottom)

(format:
  ### <date> - job-NN/step-MM
  built: ...
  checks: GREEN|RED - <one-line evidence>
  commits: <hashes>
  next: <what the next session should do first>)

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

### 2026-08-25 - job-00/step-01
built: scripts/check_env.py (env verification script)
checks: GREEN - rmgpu env has all deps; torch.cuda.is_available() True; rmgdb importable; all five SQLite DBs present; checkpoint example_model_v2_regression_mol.ckpt present and loads with chemprop example pattern (prediction for ethane CC = 2.166739)
commits: d132f91
next: job-00/step-02
