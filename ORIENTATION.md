# ORIENTATION - read once per job (first step), never per step

You are implementing RMG-GPU (rmgpu): a ground-up, pure-Python rewrite of RMG that
(1) relies on external packages to the greatest extent possible and (2) leverages GPUs
(PyTorch) for property prediction (ML models) and simulation (DAE solver, master
equation). This brief is self-contained; you do NOT need any prior conversation.

## The one-paragraph version

RMG grows a chemical reaction mechanism by iterating: estimate kinetics/thermo for
candidate reactions, simulate in a reactor, screen species by conversion, grow/prune
the model, repeat until steady. RMG-GPU keeps that orchestration and the reaction
*generation* machinery, but replaces ALL property estimation with ML models
(CheMeleon thermo, Chemprop kinetics) and replaces the reactor/pressure-dependence
numerics with PyTorch (torchdae DAE solver; master equation in torch). Everything else
(graph ops, database I/O, units, transport, interop) is delegated to packages. No
Cython, no numba, no fallbacks, no QM, no Arkane.

## The ML estimators are NEW (replacement, not reuse) -- and the checkpoints are REAL

RMG-Py already contains a Chemprop-based thermo estimator
(`rmgpy/ml/estimator.py`, an `MLEstimator` wrapping Chemprop species models; disabled
on py3.11, upstream issue #2559). In rmgpu it is REPLACED:

- rmgpu's estimators (`rmgpu/ml/thermo_estimator.py`, `rmgpu/ml/kinetics_estimator.py`)
  are written FRESH, wrapping the two vendored checkpoints (see below) with the
  inference pattern of `/home/jackson/rmgpu/chemprop_example/predicting.ipynb` and,
  more directly, this repo's own `models/predict.py` (the model team's inference code):
  load the checkpoint, build a chemprop dataset with the model's featurizer,
  `data.build_dataloader`, `pl.Trainer(...).predict`.
- RMG's `ml/estimator.py` is read ONLY as reference: its uncertainty-cutoff concepts and
  how the `ml_estimator:` input DSL wires checkpoint names. Nothing is imported, copied,
  or wrapped from it. (Its two-checkpoint Hf298/S298+Cp layout does NOT match our
  checkpoints -- the vendored thermo model is ONE model with 9 outputs; see 3b.)
- **The checkpoints are vendored in this repo's top-level `models/` directory** (committed,
  copied inference-only from the separate `ml-fitting` repo where the models are trained):
  - `models/chemeleon_thermo_122e91.ckpt` -- CheMeleon MPNN; targets
    `log_H298_J_mol, log_S298_J_mol_K, log_Cp_1..7_J_mol_K` (log10-space; Cp at
    300/400/500/600/800/1000/1500 K); featurizer `SimpleMoleculeMolGraphFeaturizer`;
    trained on 1662 rmgdb thermo-library species.
  - `models/chemprop_kinetics_122e91.ckpt` -- Chemprop reaction model; targets
    `log10_A, n, Ea_J_mol` (A = per-site pre-exponential in cm^3/(mol*s); Ea linear
    J/mol); RIGR featurizer (`CondensedGraphOfReactionFeaturizer`); input = atom-mapped
    reaction SMILES; trained on rmgdb kinetics-library HPL params.
  - `models/predict.py` (predictor classes), `models/config.py` (target names),
    `models/models.py` (inference-only model defs).
- **Load-path constraint:** the checkpoints pickle-reference
  `models.BoundedOutputTransform` and `models.HuberMetric` -- a module importable as
  top-level `models` (i.e. `models/models.py`) is REQUIRED to load them; do not rename
  or move it, and do not alter those classes.
- **Boundary conversions** (raw -> SI, PLAN.md 3b): thermo 10^pred (H298 J/mol, S298
  J/mol/K, Cp 7 grid points J/mol/K); kinetics A = 10^pred * degeneracy (CGS), n as-is,
  Ea as-is (J/mol). The model's Cp outputs are 7 discrete grid values; the estimator
  wraps them into rmgpu/data/thermo's Cp(T) representation.
- A future checkpoint (better model) is a config change (`ml_estimator:` block:
  name + hash -> file in `models/`), never a code change (PLAN.md 8a.3: the checkpoint
  interface is the only seam between model improvement and mechanism generation).
- Training/fitting stays OUT of scope: the `ml-fitting` repo owns model development;
  rmgpu only ever loads and predicts (PLAN.md 8a.3).

## What stays, what goes (the retain/delete/external split)

RETAIN (port from RMG-Py, modernize to numpy/torch; this is the actual IP):
  1. Core/edge mechanism loop          RMG-Py/rmgpy/rmg/model.py
  2. Reaction recipe DSL               RMG-Py/rmgpy/data/kinetics/family.py
     (apply_recipe, product generation, degeneracy) -- custom atom-labeled bond-ops
     language, NOT SMARTS reactions
  3. Resonance generation              RMG-Py/rmgpy/molecule/resonance.py
  4. Atom-type DB + assignment         RMG-Py/rmgpy/molecule/atomtype.py
  5. Master equation / pdep            RMG-Py/rmgpy/pdep/network.py + collision models
  6. Statmech                          RMG-Py/rmgpy/statmech/*
  7. Chemkin I/O (thin) + Cantera export

DELETE (do NOT port):
  - Group additivity + HBI thermo      (RMG-Py/rmgpy/data/thermo.py group-adding code)
  - Rate rules + Bayesian-modeling tree (rmgpy/data/kinetics/rules.py, tree parts of family.py)
  - The multi-method fallback chains in thermo/kinetics lookup
  - QM (rmgpy/qm) -- entirely out of scope
  - Arkane dependency -- nothing mainlined
  - RMG's own Chemprop wrapper (rmgpy/ml/estimator.py) -- REPLACED by rmgpu/ml/
    (see "The ML estimators are NEW" above)

EXTERNAL (never re-implement):
  - RDKit: molecule graph, isomorphism/substructure (degeneracy counting),
    canonicalization, SMARTS, structure drawing (rdMolDraw2D)
  - rmgdb (/home/jackson/rmgpu/rmgdb): ALL database I/O. SQLAlchemy + SQLite + YAML.
    DBs: thermo.db, kinetics.db, transport.db, solvation.db, statmech.db (+ surface).
    Read its README.md and the schemas in standard/rmgdb/{thermo,kinetics,transport,solvation,statmech}/schema.py
    before writing any loader.
  - torch + torchdae: reactor (sole DAE backend; BDF/TR-BDF2/Radau-IIA; GPU via .to(cuda));
    master-equation time integration; adjoint sensitivity
  - chemprop + lightning: the two vendored checkpoints in models/ (PLAN.md 3b):
    thermo = CheMeleon MPNN molecule model (9 log-space targets);
    kinetics = RIGR reaction model (log10_A, n, Ea). Load/predict pattern in models/predict.py.
    models/models.py must stay importable as top-level `models` (checkpoint load constraint).
  - chemicals/fluids/thermo (ChEDL): transport correlations
  - pint: units (replaces the cimported Quantity)
  - cantera: ck2yaml interop (RMG already uses it)

## The master equation data path (important; no QM anywhere)

The pdep engine needs, per isomer/TS: E0, vibrational frequencies (for the density of
states), rotor barriers, collision parameters. In rmgpu ALL of these come from:
  - E0 (isomers): ML thermo Hf298
  - frequencies/rotors: the STATMECH DATABASE (group characteristic frequencies, fit
    to Cp) -- see RMG-Py/rmgpy/data/statmech.py::get_statmech_data (lines ~318-460)
    for the algorithm to port. No QM.
  - TS E0: DERIVED FROM THE HPL RATE: E0_TS = sum(E0_reactants) - R*T*ln(k_inf*V/h)
    (see /home/jackson/rmgpu/RMG-Py/arkane/pdep.py lines ~273-282 for RMG's version).
    The HPL rate is the ML-predicted kinetics. No QM.
  - collision (LJ sigma/eps): transport DB via rmgdb.
This is the single most misunderstood part of the design; if in doubt, re-read
PLAN.md section 8a.

## Conventions

  - Python 3.11+, conda env `rmgpu` (see job 00). GPU: local CUDA box; single heavy
    GPU task at a time.
  - No Cython, no numba (unless a specific kernel is proven hot and approved).
  - No fallbacks: ML models are the only estimators; torchdae is the only reactor
    backend. Gaps in ML coverage are RECORDED FINDINGS, not silent fallbacks.
  - Types: quantity = (value, unit) via pint; SI internally (J, K, Pa, mol).
  - Molecule: thin wrapper over RWMol carrying: atom-type assignments, RMG labels,
    resonance structure list, charge/radical count. Canonicalization and isomorphism
    go through RDKit. The RMG adjacency list is a FORMAT (parse/serialize), not the
    internal representation.
  - Reference behavior always comes from RMG-Py run on the same machine, captured
    into gates/baselines. When rmgpu and RMG-Py disagree, the gate report says so.
  - Interpreter: always the env's python directly,
    /home/jackson/miniforge3/envs/rmgpu/bin/python. Never install into system
    Python or the active venv.

## The database is the moat

RMG-database (2.4M lines: 135 families, 74 kinetics libraries, 77 thermo libraries,
statmech/transport/solvation/surface) is untouched and read through rmgdb. Do not
modify it. Round-trip/hash gates (job 02) guard against I/O drift.

## How work proceeds: jobs and steps (read this)

Two levels (full protocol in README.md, "Session vs step"):

  JOB   = a milestone with one final gate (prompts/job-NN-*.md = the brief:
          goal + step list + gate definition). A job is TOO BIG for one session;
          never attempt to do a job in a single session.
  STEP  = one self-contained unit of work sized for ONE fresh human-started session
          (prompts/steps/job-NN-step-MM-*.md). Each step file names exactly which
          reference files to read (with a context budget), what to build, the step's
          checks, and the done protocol.

The loop (run by the human, one session at a time):
  1. STATUS.md's NEXT pointer names exactly one step.
  2. A fresh session reads that step file (plus what it points at) and does
     ONLY that step. Sessions do not spawn subagents.
  3. The step commits ("job-NN/step-MM: ..."), writes reports/job-NN-step-MM-*.md,
     and updates its step row in STATUS.md.
  4. The human reviews, then sets NEXT to the following step (or a fix step
     if the checks came back red).
  5. The job's final step runs the job gate (gates/gate_NN.py -> reports/job-NN.md);
     only a GREEN gate (or a documented, user-accepted finding) closes the job.

Why steps: a job pulls 20-150k lines of reference code through one context; that is
what was corrupting long sessions (context compaction + forgetting). Each step reads
at most ~3k lines of reference (its file says the budget), so a session finishes
without compaction. Steps in a job are strictly sequential: a step's output files are
the next step's input.

This file (ORIENTATION.md) is read ONCE, by the first step of a job. Later steps of
the same job do not need it - their step file repeats the few facts they require.

## Files you will create (structure from job 00)

    rmgpu/
      core/        model.py recipe.py atomtype.py resonance.py template.py family.py
      molecule/    molecule.py adjlist.py atomtype.py resonance.py symmetry.py
                   filtration.py group.py
      db/          loaders.py
      ml/          base.py thermo_estimator.py kinetics_estimator.py
      kinetics/    models.py
      pdep/        network.py collision.py msc.py rs.py sls.py driver.py
      statmech/    (modes, torsion, assembly)
      data/        entries.py thermo.py kinetics.py statmech.py estimation.py
      reactor/     torch.py reactors.py
      transport.py io/  units.py  input.py  main.py  output.py
      schemas/     (pydantic: input + mechanism artifact)
      tools/       isotopes.py observables.py diffmodels.py mergemodels.py
      sensitivity/ (sensitivity.py uncertainty.py)
      plugins/     base.py solvation/ catalysis/
    gates/         gate_NN.py scripts
    reports/       job-NN.md + job-NN-step-MM-*.md + parity/
    tests/         pytest suites (unit + integration)

## If something in a prompt contradicts PLAN.md

PLAN.md is the source of design truth. Prompts (job briefs + step files) are the
task decomposition. If they conflict, note it in STATUS.md and follow PLAN.md unless
the user has directed otherwise in STATUS.md.
