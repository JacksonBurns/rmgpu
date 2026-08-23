# job-06/step-02: CoreEdgeReactionModel (enlarge/prune/screen)

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

- Previous step (already done, its code is in the tree): job-06/step-01-reactor
- This step: job-06/step-02-model
- Next step (do NOT start it): job-06/step-03-driver

## Goal

Port the core/edge mechanism loop: CoreEdgeReactionModel - the heart
of the product. core/edge bookkeeping, enlarge (generate + estimate +
simulate + screen), prune, thermo filter, edge->core promotion.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/rmg/model.py  (2334 - CoreEdgeReactionModel:
    enlarge, prune, thermo_filter_species, add_reactions_to_model,
    generate_reactions, edge->core promotion, the whole bookkeeping -
    port the logic EXACTLY; the parity gate depends on it)
  RMG-Py/rmgpy/rmg/react.py  (172 - the reaction generation driver)
  RMG-Py/rmgpy/species.py + reaction.py (the Species/Reaction objects:
    what the model holds - port the data semantics, not the Cython)
  (job-04's resolvers (estimation.py), job-05's families, job-02's DBs)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/core/model.py: `CoreEdgeReactionModel`:
    core/edge sets (species + reactions, with rates), `enlarge()` (generate
    candidate reactions via job-05 families + templates, estimate
    kinetics/thermo via job-04 resolvers, simulate the current model via
    job-06/1's reactor backend, screen species by conversion per RMG's
    rules), `prune()` (rate/thermo-based removal per RMG's rules - the
    filter_reactions logic), thermo_filter_species (per RMG), edge->core
    promotion by conversion threshold (the model block's tolerances:
    tolerance_move_to_core, tolerance_keep_in_edge,
    tolerance_interrupt_simulation - from the YAML, job 03).
    The reaction object unifies species + rate model (job 02/04 registry)
    + family/template provenance (job 05).
    The pdep HOOK: reactions in pressure-dependent families are marked;
    until job 07 lands, they get HPL-only kinetics (a documented stub -
    this keeps the job honest and gas-phase HPL; job 07 replaces the stub
    without touching the loop).
- tests/test_core_model.py: a TINY mechanism (3 species, 2-3 reactions)
  hand-built (no DB): enlarge generates the expected candidates; screen
  promotes/demotes by conversion threshold (a contrived profile); prune
  removes a low-rate reaction per the rule; the bookkeeping invariants
  (every reaction's species are in core/edge; rates are set) hold after
  each op.
- A `Mechanism` value object (species + reactions + provenance) - the
  thing the output tree (job 06/5) serializes and the seed loader (job
  08/10) deserializes - defined here (dataclass-ish, plain data).

## Checks (must run and pass before you claim done)

  pytest tests/test_core_model.py -q -> all pass
  invariants hold on the tiny mechanism after enlarge/prune/promote

## Pitfalls

- The core/edge bookkeeping is intricate (edge reactions, reaction
  rates for screening, the "interrupt" tolerance) - port RMG's logic
  exactly; do not "simplify" a bookkeeping rule - the parity gate (step 5)
  is the spec.
- Screening uses reaction RATES at the current conditions - make sure the
  rate registry's k(T,P) is callable per (T,P) fast enough for a loop (no
  per-iteration Python loops over reactions if avoidable - but correctness
  first, profile later; do not micro-optimize prematurely).
- MLCoverageError from the resolvers (job 04): the loop must CATCH it,
  record the species/reaction as a coverage finding (the run's summary
  shows the split), and continue - never crash, never fall back.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-06/step-02: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-06/step-02 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-06-step-02-model.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
