# job-06/step-06: Job-06 gate (first real mechanism generation)

Job: job-06 - Core/edge mechanism loop + torchdae reactor (first integration)
Prereq: jobs 01-05 done (all of: molecule, db, schema, estimators, recipes)
This step is part of that job. The job's overall goal:
The `rmgpu run` command actually works end to end (gas phase, constant T or T/P): load YAML -> build model from seed species -> estimate properties (libraries/ML) -> enumerate candidate reactions (recipes) -> simulate in torchdae -> screen by conversion -> grow/prune -> iterate to steady state -> write the output tree (PLAN.md 12.3), including the canonical `mechanism/core.yaml`. The moment of truth for the stack.
The job's gate (run by the job's final step):
gates/gate_06.py: 1. torchdae sub-gate: a stiff reference ODE (a 3-reaction Lindemann falloff toy system, or Van der Pol mu=10 as a non-chemistry control) integrated by torchdae vs a high-accuracy reference (RK45 tiny step) - max abs diff recorded (PLAN.md 13 risk 3: prove it before trusting it with mechanism growth). 2. `rmgpu run` on the imported `superminimal` (HPL kinetics, pdep off/stub): completes to steady state (or max_iter); iteration count + core/edge species+reaction counts recorded. 3. Parity vs RMG-Py (pdep OFF, same input; its output -> gates/baselines/superminimal/): |core_rmgpu - core_rmg| and |core+edge_rmgpu - (core+edge)_rmg| as sets. TARGET: core identical (or documented divergence with cause); edge within a small fraction. Any systematic divergence (a family always missing/extra): fix if tractable in this job, else record precisely. 4. Same for `c3h4` (the right size - NOT the GRI-scale example). 5. Output tree: every file in PLAN.md 12.3 exists and is valid (YAML parses, CSV columns right, core.yaml loads back via the schema and re-simulates the final iteration's profiles within tolerance). 6. provenance.yaml contains real hashes/versions (not placeholders).

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

- Previous step (already done, its code is in the tree): job-06/step-05-chemkin
- This step: job-06/step-06-gate
- Next step (do NOT start it): the job gate for job-06 (its step file)

## Goal

Run the job-06 gate: torchdae sub-gate, superminimal + c3h4 end-to-
end with parity vs RMG-Py, the output tree, provenance. The first real
mechanism generation - the moment of truth.

## Reference to read (this step's budget)

  (no new reference reads)
  examples: /home/jackson/rmgpu/RMG-Py/examples/rmg/{superminimal,c3h4}/
    (their input.py - imported to YAML via job-03; a modified copy with
    pdep OFF for the RMG-Py reference run)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- gates/gate_06.py:
    (1) the torchdae stiff-ODE sub-gate (re-run; max abs diff vs the
    reference);
    (2) `rmgpu run` on the imported superminimal (pdep off/stub):
    completes; iteration count + core/edge counts;
    (3) parity vs RMG-Py (its output -> gates/baselines/superminimal/):
    core species set + core reaction set (canonical SMILES keys) - the
    set diffs (species/reactions on each side); edge likewise;
    (4) same for c3h4;
    (5) the output tree: every file in PLAN.md 12.3 exists + valid;
    core.yaml loads back + re-simulates the final iteration's profiles
    within tolerance;
    (6) provenance: real hashes/versions.
- The RMG-Py reference: a script (scripts/rmgpy_reference.py) that runs
  RMG-Py (env rmg_env) on the modified inputs (pdep OFF - match rmgpu's
  HPL-stub state) into gates/baselines/<example>/ (commit them).
- reports/job-06.md: the sub-gate numbers, the per-example iteration
  counts + core/edge parity (the set diffs, listed), the systematic-
  divergence analysis (a family always missing/extra? an estimator
  difference? - root-caused, each with the responsible module), the
  blocked-families impact (from job 05), the coverage-findings count
  (ML gaps recorded).
- A RED parity (core not identical with no documented cause): fix the
  responsible module (job 04/05/06) if tractable in this step - re-run;
  else record precisely (the example + the diff + the suspected module)
  in STATUS + the report.

## Checks (must run and pass before you claim done)

  /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_06.py
    -> the numbers recorded (core parity target: identical or documented;
    edge within a small fraction)
  pytest tests/ -q -> all pass

## Pitfalls

- The parity gate's reference is RMG-Py with pdep OFF - matching
  rmgpu's HPL-stub state (job 07 turns pdep on in BOTH). A pdep-on vs
  pdep-off comparison is a category error.
- Systematic divergences (one family always missing/extra) are the
  signal - fix the cause in the responsible module, not the example.
- Do NOT tune the model tolerances to force agreement - the imported YAML
  carries RMG's exact values (job 03).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-06/step-06: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-06/step-06 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-06-step-06-gate.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
