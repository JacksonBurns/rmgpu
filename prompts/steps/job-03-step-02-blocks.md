# job-03/step-02: Input schema: reactors + remaining blocks + extends

Job: job-03 - YAML input schema + CLI + legacy .py importer
Prereq: job-01 done (Molecule for structure parsing). job-02 not strictly required, but the database: block references library names from it.
This step is part of that job. The job's overall goal:
Replace RMG's "execute a Python file to configure the run" with a declarative, schema-validated YAML document + a small CLI + a lossless importer for the legacy .py DSL. This is the user-facing front door. Deliverables: rmgpu/schemas/input.py (pydantic), rmgpu/cli.py (click), rmgpu/importer/legacy.py (ast-based).
The job's gate (run by the job's final step):
gates/gate_03.py: 1. DSL inventory: 47 legacy input.py files x functions used (report table). 2. Import: N of 47 files import to schema-valid YAML with zero dropped values; the rest have documented IMPORT-NOTEs. Target: N == 47. 3. `rmgpu validate` on all 47 imported YAMLs: all pass. 4. `rmgpu run minimal.yaml` (the imported minimal example) prints the resolved document. 5. JSON schema exports and validates a hand-written minimal input.yaml (the example from PLAN.md 12.2).

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

- Previous step (already done, its code is in the tree): job-03/step-01-core
- This step: job-03/step-02-blocks
- Next step (do NOT start it): job-03/step-03-cli

## Goal

Schema part 2: the polymorphic reactors block + simulator, model,
pressure_dependence, ml_estimator, solvation, uncertainty, options - and
extends resolution as a loadable feature.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/rmg/input.py (the reactor DSL functions:
    simpleReactor, constant_V_ideal_gas_reactor,
    constant_T_P_ideal_gas_reactor, liquid_reactor, mb_sampled_reactor,
    surface_reactor, staged reactors; simulator; model; pressure_dependence;
    uncertainty; solvation; ml_estimator; options - signatures + semantics)
  /home/jackson/rmgpu/rmgpu/PLAN.md 12.2 (the example yaml: the reactor
    types, termination, the method strings)
  RMG-Py/rmgpy/solver/termination.py (66 - the termination conditions:
    conversion, time, criticality)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/schemas/input.py (extend):
    Reactors: polymorphic on `type` (simple | const_V | const_TP | liquid |
    mb_sampled | surface) - each its own typed model; surface gets a typed
    model NOW even though the surface plugin lands in job 12 (input must not
    break); staged reactors (nested lists) where the DSL has them;
    termination (conversion {species, value}, time, criticality).
    SimulatorBlock {atol, rtol}; ModelBlock (tolerance_keep_in_edge,
    tolerance_move_to_core, tolerance_interrupt_simulation,
    maximum_edge_species, filter_reactions); PressureDependenceBlock
    (method: cse|masc|rs|sls shorthand OR the long strings - the
    normalization table from job 07's note: 'modified strong collision' etc.;
    Tmin/Tmax/Tcount, Pmin/Pmax/Pcount, grain controls, interpolation_model);
    MLEstimatorBlock {thermo: checkpoint ref, kinetics: checkpoint ref};
    SolvationBlock {solvent, model: smd}; UncertaintyBlock {enabled,
    species}; OptionsBlock (save_profiles, save_plots, save_edge, units).
    Top-level Input model completes (all blocks).
- extends: `load_input(path)` - resolve extends chains into one flat
  document (earliest wins), cycle detection, file-relative path resolution.
- tests/test_schemas_blocks.py: polymorphic dispatch on type; a staged
  reactor; pressure_dependence method normalization (shorthand + long
  strings); extends 2-level + cycle; the PLAN.md 12.2 example validates
  end-to-end.

## Checks (must run and pass before you claim done)

  pytest tests/test_schemas_blocks.py -q -> all pass

## Pitfalls

- The method-string normalization must accept BOTH forms (job 07's
  driver dispatches on the long strings; the YAML user may type either).
- Keep surface/liquid/mb_sampled typed NOW - a later "the input schema
  broke when we added the plugin" is a schema design failure, not a plugin
  bug.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-03/step-02: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-03/step-02 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-03-step-02-blocks.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
