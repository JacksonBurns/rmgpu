# STATUS - rmgpu work tracker

Single source of truth for work state across sessions. Every session updates this
file before committing. Do not delete entries; append and annotate.

## NEXT (the pointer - the human reads this first)

NEXT: prompts/steps/job-03-step-04-legacy.md (job-03 step-03 done: CLI run/validate/schema/version, 7 tests pass)
(When a step finishes, the session updates this pointer to the following
step's file, or to a small fix-step file written for a red gate. One step
at a time.)

## Job table (a job is done only when its GATE step is GREEN)

| Job | Title | Gate | Status |
|-----|-------|------|--------|
| 00 | Env + package skeleton + test scaffolding | smoke test (gate_00.py) | done |
| 01 | Units + molecule layer | adjlist/atomtype/resonance parity (gate_01.py) | done |
| 02 | Database layer via rmgdb + round-trip | entry-count + table hash vs RMG-Py (gate_02.py) | done |
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

### 2026-08-27 - job-03/step-01
built: rmgpu/schemas/input.py (Quantity, StructureValue, DatabaseBlock, Species, ForbiddenEntry, Input, resolve_extends); tests/test_schemas_core.py (20 tests)
checks: GREEN - pytest tests/test_schemas_core.py -q: 20 passed
commits: 4ae76c8
next: job-03/step-02-blocks

### 2026-08-27 - job-03/step-02
built: rmgpu/schemas/input.py extended (Reactors polymorphic Union, StagedReactor, LiquidStagedReactor, ConstantVStagedReactor, PressureStagedReactor, SimulatorBlock, ModelBlock, PressureDependenceBlock, MLEstimatorBlock, SolvationBlock, UncertaintyBlock, OptionsBlock); rmgpu/units.py fixed (_coerce_quantity regex for exponential notation); tests/test_schemas_blocks.py (25 tests)
checks: GREEN - pytest tests/test_schemas_blocks.py -q: 25 passed
commits: 7cec009
next: job-03/step-03-cli
