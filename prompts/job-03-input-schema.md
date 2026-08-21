# job-03: YAML input schema + CLI + legacy importer

Read ORIENTATION.md and PLAN.md section 12 (I/O redesign) in full. Prereq: job-01.
(job-02 not strictly required, but the `database:` block references library names.)

## Goal

Replace RMG's "execute a Python file to configure the run" with a declarative,
schema-validated YAML document + a small CLI + a lossless importer for the legacy
`.py` DSL. This is the user-facing front door of the package.

## Reference (read)

  RMG-Py/rmgpy/rmg/input.py  -- the 41-function legacy DSL. The importer must
  cover ALL of them (or explicitly reject with a clear error, recorded).
  PLAN.md 12.1-12.5          -- the design: principles, the example input.yaml,
                                quantity syntax, extends/layering, CLI, output tree.
  examples: /home/jackson/rmgpu/RMG-Py/examples/rmg/*/input.py  (33 example inputs;
             47 total incl. test/regression - enumerate and inventory them first:
             for each, list which DSL functions it uses; this inventory is part of
             the deliverable and lives in reports/job-03.md).

## Deliverables

1. `rmgpu/schemas/input.py` -- pydantic models, versioned (`rmgpu: 1.0` top key):
   top-level keys: database, species, forbidden, reactors, simulator, model,
   pressure_dependence, ml_estimator, solvation, uncertainty, options, extends.
   - Quantity type: {value, unit} map OR "1350 K" string (pint); SI-internal.
   - species: list of {label, reactive, structure (smiles/inchi/adjlist-str),
     optional per-species thermo/kinetics overrides, constraints}.
   - reactors: polymorphic on `type` (simple | const_V | const_TP | liquid |
     mb_sampled | surface) - surface gets a model now (typed) even though the
     surface plugin lands in job 12 (input must not break).
   - every field gets a docstring; the models export a JSON schema
     (`rmgpu schema --out schema.json`) for editor/LSP use.
   - `extends`: resolve a chain into one flat document (earliest wins on conflicts),
     with cycle detection.
2. `rmgpu/cli.py` (click):
   - `rmgpu run input.yaml`   -- load, validate, (for now) print the resolved
     document as YAML and exit 0. Actual execution is job 06+.
   - `rmgpu validate input.yaml` -- validate only, report ALL problems at once
     (no fail-fast), exit 0/1.
   - `rmgpu import old.py --to new.yaml` -- the legacy importer (see 3).
   - `rmgpu schema --out path.json`
   - `rmgpu version`
   (diff/export/inspect are later jobs - stub with "not yet implemented".)
3. `rmgpu/importer/legacy.py` -- parse legacy Python DSL with `ast` (NEVER exec):
   - a visitor that maps each of the 41 DSL functions to schema fields.
   - handles: SMILES("/InChI/adjacency_list constructors, dict-of-dicts initialMoleFractions,
     (value, unit) tuples -> Quantity, 'auto'/sentinel values, nested lists
     (staged reactors), keyword defaults.
   - anything it cannot express: fail loudly with a message naming the function +
     line number, and write it to a `# IMPORT-NOTE:` comment block at the top of the
     output YAML. Never silently default.
4. `tests/test_importer.py`: for every one of the 47 legacy input.py files:
   import to YAML, re-parse the YAML against the schema (must validate), and diff
   the resolved document against a ground-truth dump of the legacy file's values.
   Ground truth: a small `legacy_dump.py` that uses ast (same visitor, no schema) to
   emit the raw parsed structure as JSON; the gate compares structure equality
   (values only; ordering where the DSL is order-sensitive must be preserved).
   Files the importer cannot fully handle: allowed ONLY if documented with an
   IMPORT-NOTE and the schema-validatable subset still matches; list every such file
   in the report.

## Gate (job 03) -> gates/gate_03.py, report reports/job-03.md

  1. DSL inventory table: 47 files x functions used (in the report).
  2. Import: N of 47 files import to schema-valid YAML with zero dropped values;
     the rest have documented IMPORT-NOTEs. Target: N == 47 (this is the point of
     the job).
  3. `rmgpu validate` on all 47 imported YAMLs: all pass.
  4. `rmgpu run minimal.yaml` (the imported minimal example) prints the resolved doc.
  5. JSON schema exports and validates a hand-written minimal input.yaml
     (the example from PLAN.md 12.2).

## When done

STATUS.md update + commit "job-03: YAML schema + CLI + legacy importer" + STOP.

## Pitfalls

- The legacy DSL has implicit Python semantics (default args evaluated in the .py,
  `auto` strings, nested tuples). The importer must pin them explicitly.
- `restart_from_seed` and seedMechanisms point at files; the schema keeps them as
  paths (resolved relative to the input file's directory).
- Do not wire `rmgpu run` into execution yet - validation + resolution only.
- Keep pydantic models importable without heavy deps where possible (the schema is
  the API; PLAN.md 12.4).
