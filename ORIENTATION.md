# ORIENTATION - read this first, every session

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

EXTERNAL (never re-implement):
  - RDKit: molecule graph, isomorphism/substructure (degeneracy counting),
    canonicalization, SMARTS, structure drawing (rdMolDraw2D)
  - rmgdb (/home/jackson/rmgpu/rmgdb): ALL database I/O. SQLAlchemy + SQLite + YAML.
    DBs: thermo.db, kinetics.db, transport.db, solvation.db, statmech.db (+ surface).
    Read its README.md and the schemas in standard/rmgdb/{thermo,kinetics,transport,solvation,statmech}/schema.py
    before writing any loader.
  - torch + torchdae: reactor (sole DAE backend; BDF/TR-BDF2/Radau-IIA; GPU via .to(cuda));
    master-equation time integration; adjoint sensitivity
  - chemprop: kinetics prediction (reaction mode: RxnMode enum, see PLAN.md 6)
  - CheMeleon checkpoints: thermo prediction (Hf298, S298, Cp(T))
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

## The database is the moat

RMG-database (2.4M lines: 135 families, 74 kinetics libraries, 77 thermo libraries,
statmech/transport/solvation/surface) is untouched and read through rmgdb. Do not
modify it. Round-trip/hash gates (job 02) guard against I/O drift.

## How a job session proceeds (repeat this)

  1. Read this file, then the job prompt (prompts/job-NN-*.md), then STATUS.md
     (tail: last session log entry + the job table).
  2. Work the job: read the named RMG-Py reference files, implement, test.
  3. Run the job's gate (gates/gate_NN.py or as specified). Write the report to
     reports/job-NN.md with REAL output (counts, diffs, timings). If the gate is red,
     say so plainly in the report; fix what you can in the session; leave the rest
     as explicit TODOs in STATUS.md.
  4. Update STATUS.md: flip the job row (in-progress -> done|blocked), append a
     session-log entry: date, what was built, gate result (GREEN/RED + one-line
     evidence), commit hashes, next-step notes for the following session.
  5. Commit (message: job-NN: <summary>). Do NOT push. Do NOT start the next job.
  6. Stop and report.

## Files you will create (structure from job 00)

    rmgpu/
      core/        model.py recipe.py atomtype.py resonance.py
      molecule/    molecule.py adjlist.py
      db/          loaders.py
      ml/          thermo_estimator.py kinetics_estimator.py
      kinetics/    models.py
      pdep/        network.py collision.py
      statmech/    ...
      reactor/     torch.py reactors.py
      transport.py io/  units.py  input.py  main.py  output.py
      schemas/     (pydantic: input + mechanism artifact)
    gates/         gate_NN.py scripts
    reports/       job-NN.md
    tests/         pytest suites (unit + integration)

## If something in a prompt contradicts PLAN.md

PLAN.md is the source of design truth. Prompts are the task decomposition. If they
conflict, note it in STATUS.md and follow PLAN.md unless the user has directed
otherwise in STATUS.md.
