# job-06/step-04: Mechanism artifact schema + the output tree writer

Job: job-06 - Core/edge mechanism loop + torchdae reactor (first integration)
Prereq: jobs 01-05 done (all of: molecule, db, schema, estimators, recipes)
This step is part of that job. The job's overall goal:
The `rmgpu run` command actually works end to end (gas phase, constant T or T/P): load YAML -> build model from seed species -> estimate properties (libraries/ML) -> enumerate candidate reactions (recipes) -> simulate in torchdae -> screen by conversion -> grow/prune -> iterate to steady state -> write the output tree (PLAN.md 12.3), including the canonical `mechanism/core.yaml`. The moment of truth for the stack.
The job's gate (run by the job's final step):
gates/gate_06.py: 1. torchdae sub-gate: a stiff reference ODE (a 3-reaction Lindemann falloff toy system, or Van der Pol mu=10 as a non-chemistry control) integrated by torchdae vs a high-accuracy reference (RK45 tiny step) - max abs diff recorded (PLAN.md 13 risk 3: prove it before trusting it with mechanism growth). 2. `rmgpu run` on the imported `superminimal` (HPL kinetics, pdep off/stub): completes to steady state (or max_iter); iteration count + core/edge species+reaction counts recorded. 3. Parity vs RMG-Py (pdep OFF, same input; its output -> gates/baselines/superminimal/): |core_rmgpu - core_rmg| and |core+edge_rmgpu - (core+edge)_rmg| as sets. TARGET: core identical (or documented divergence with cause); edge within a small fraction. Any systematic divergence (a family always missing/extra): fix if tractable in this job, else record precisely. 4. Same for `c3h4` (the right size - NOT the GRI-scale example). 5. Output tree: every file in PLAN.md 12.3 exists and is valid (YAML parses, CSV columns right, core.yaml loads back via the schema and re-simulates the final iteration's profiles within tolerance). 6. provenance.yaml contains real hashes/versions (not placeholders).

## Context (invariant every step)

- You are a FRESH subagent session; you have no other context. This file plus
  what it points at is everything you need.
- Work in /home/jackson/rmgpu/rmgpu (git, branch main). Reference repos (read-
  only, at /home/jackson/rmgpu): RMG-Py, RMG-database, rmgdb, chemprop_example.
- Interpreter: always the env's python directly,
  /home/jackson/miniforge3/envs/rmgpu/bin/python (created in job 00).
  NEVER install into system python or the active venv.
- ONE subagent at a time; this session MUST NOT spawn subagents. Single heavy
  GPU task at a time on this machine; never kill processes you did not start.
- No Cython, no numba, no QM, no Arkane, no fallback estimators, no second
  reactor backend. SI units internally (J, K, Pa, mol) via pint.
- The existing Chemprop-based estimators in RMG-Py (rmgpy/ml/estimator.py)
  are being REPLACED by rmgpu's new ones - they are reference-only (layout,
  cutoffs, DSL wiring). rmgpu's estimators are re-implemented per
  /home/jackson/rmgpu/chemprop_example/predicting.ipynb. See README.md,
  "The ML estimators are NEW".

## Where you are

- Previous step (already done, its code is in the tree): job-06/step-03-driver
- This step: job-06/step-04-output
- Next step (do NOT start it): job-06/step-05-chemkin

## Goal

The canonical artifact: the mechanism schema (core.yaml - bidirectional
seed) and the full output tree per PLAN.md 12.3 (the writer every later
job extends).

## Reference to read (this step's budget)

  /home/jackson/rmgpu/rmgpu/PLAN.md 12.3 (the output tree - the spec)
    + 12.4 (the schema is shared & versioned)
  (job-03's schema conventions: versioned pydantic, the rmgpu: 1.0 key)
  RMG-Py/rmgpy/rmg/output.py + species/reaction serialization (READ THE
    DATA FLOW ONLY: what a mechanism's species/reactions carry that the
    artifact must hold - source, method, uncertainties)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/schemas/mechanism.py: the canonical artifact schema
  (pydantic, versioned):
    Mechanism: version key, species (each: label, formula, smiles,
    adjlist, source (library name / ml / depository), symmetry, thermo
    {model: wilhoit|nasa7|ml, Hf298, S298, Cp(T) grid or coeffs, method:
    ml|library, uncertainty}), reactions (each: label, reactants,
    products, family, template, source, degeneracy, rate model {type +
    params, method: ml|library, uncertainty}), core/edge split.
    Loadable back into a Mechanism object (bidirectional seed - the
    `database.seed_mechanisms` path reads it; the loader function lands
    here, the wiring into main.py is job 08/10).
    Exportable: to_dict for the writers (step 6's chemkin writer; job
    08/8's cantera export).
- rmgpu/output.py: the output-tree writer per PLAN.md 12.3 (extensible -
  the pdep/uncertainty/iterations dirs are added by their jobs; make the
  writer a registry of per-dir writers):
    run/
      run.yaml (the EXACT resolved input), provenance.yaml (rmgpu git
      hash + rmgdb version+hash + ML checkpoint hashes + torch/rdkit/
      cantera/cantera versions + python version + timestamp),
      summary.md (one page: core/edge counts, iterations, ML-vs-library
      coverage split (from job-04's resolver counts), warnings,
      coverage-findings list, provenance summary), rmgpu.log (human),
      events.jsonl (per-iteration/per-species/per-reaction events -
      structured),
      mechanism/core.yaml + edge.yaml (the canonical artifact),
      mechanism/chemkin.inp + chem_annotated.inp +
      species_dictionary.txt (step 6's writer), mechanism/cantera/ (job
      08), mechanism/rms.yaml (job 08),
      species/<label>.json (+ .svg via rdMolDraw2D),
      reactions/reactions.json,
      profiles/<reactor>/time_series.csv (columns: time[s],
      <species>_molefrac, ... - well-formed per PLAN.md 12.3) +
      metadata.yaml (conditions, termination, solver params, units).
- main.py (extend): write the output tree at run end (and per-iteration
  events to events.jsonl as the loop runs).
- tests/test_output.py: a tiny run's tree: every file exists; YAML parses;
  core.yaml loads back via the schema (round-trip: mechanism -> yaml ->
  mechanism, equal); the CSV has the right columns; provenance.yaml has
  real values (git hash resolves, not a placeholder).

## Checks (must run and pass before you claim done)

  pytest tests/test_output.py -q -> all pass
  `rmgpu run` on the minimal example: the full tree exists + validates
    (a small script in the test walks PLAN.md 12.3's file list)
  core.yaml round-trips (load -> Mechanism == the run's final model)

## Pitfalls

- core.yaml is the API (PLAN.md 13 risk 9): versioned, stable -
  the schema is the contract; a breaking change is a major version +
  migrator (do not churn it to make this step easier).
- Provenance is written EVERY time - not an option (PLAN.md 12.1.3).
- The tree is DETERMINISTIC (sorted, stable) - the regression harness
  (job 10) diffs trees; a non-deterministic tree (timestamps in filenames,
  dict-order species) breaks it.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-06/step-04: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-06/step-04 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-06-step-04-output.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
