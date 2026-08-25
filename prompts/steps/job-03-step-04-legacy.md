# job-03/step-04: Legacy importer: inventory + ast visitor

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

- Previous step (already done, its code is in the tree): job-03/step-03-cli
- This step: job-03/step-04-legacy
- Next step (do NOT start it): job-03/step-05-gate

## Goal

The ast-based legacy importer (NEVER exec): first the DSL inventory
(47 files x functions), then the visitor that maps the 41 DSL functions onto
the schema.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/rmg/input.py  (2057 - ALL 41 functions; the visitor
    must cover each or explicitly reject with a recorded error)
  The 47 legacy inputs: /home/jackson/rmgpu/RMG-Py/examples/rmg/*/input.py +
    /home/jackson/rmgpu/RMG-Py/test/regression/*/input.py (enumerate +
    inventory - which DSL functions each file uses; the inventory table is a
    deliverable in the report)
  RMG-Py/examples/rmg/minimal/input.py (the reference mapping)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/importer/legacy.py:
    `import_legacy(path) -> dict` (the schema-shaped raw document BEFORE
    pydantic validation): an ast visitor that walks the legacy .py and maps
    each of the 41 DSL functions to schema fields. Handles: SMILES()/InChI/
    adjacency_list constructors; dict-of-dicts initialMoleFractions;
    (value, unit) tuples -> Quantity; 'auto'/sentinel values; nested lists
    (staged reactors); keyword defaults; `auto` library sentinels.
    Anything it cannot express: fail loudly with a message naming the
    function + line number, and record it in an IMPORT-NOTE list (the CLI
    writes it as a `# IMPORT-NOTE:` comment block at the top of the output
    YAML). Never silently default.
- rmgpu/cli.py (extend): `rmgpu import old.py --to new.yaml` - run the
  visitor, validate against the schema (all problems), write the YAML (+
  IMPORT-NOTES).
- scripts/legacy_dump.py: a parallel ast walk (same visitor, NO schema)
  that emits the raw parsed structure as JSON - the ground truth for the
  gate's diff.
- The DSL inventory (47 files x functions) - write it to the report as it
  is built (it is part of the deliverable).

## Checks (must run and pass before you claim done)

  scripts/legacy_dump.py runs on all 47 files without crashing
    (dumps JSON to /tmp or a scratch dir - do not commit dumps)
  `rmgpu import <minimal example> --to /tmp/minimal.yaml` -> schema-valid
  (validate passes)
  pytest tests/test_importer_basic.py -q -> pass (the visitor unit tests:
  each DSL function on a synthetic snippet)

## Pitfalls

- NEVER exec/eval the legacy files - ast only (PLAN.md 12.1.1; the
  legacy files have implicit Python semantics that a schema must either
  model or reject explicitly).
- The legacy DSL has implicit Python semantics (default args, `auto`
  strings, nested tuples) - pin them explicitly in the visitor; a silently
  dropped value is a gate failure.
- Do NOT try to make all 47 pass here - the gate (step 5) is where the
  lossless target is enforced; this step lands the machinery + the obvious
  cases.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-03/step-04: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-03/step-04 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-03-step-04-legacy.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
