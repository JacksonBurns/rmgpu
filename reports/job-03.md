# Job-03: YAML input schema + CLI + legacy importer

Status: **CLOSED** - gate GREEN (`gates/gate_03.py`, exit 0), 2026-08-28.

## Goal
Replace RMG's "execute a Python file to configure the run" with a declarative,
schema-validated YAML document + a small CLI + a lossless importer for the
legacy `.py` DSL. This is the user-facing front door.

## Deliverables (all present)
- `rmgpu/schemas/input.py` - pydantic schema (lenient blocks, all blocks the
  legacy corpus uses, `StructureValue`, extends resolution).
- `rmgpu/cli.py` - `rmgpu run / validate / schema / version / import`.
- `rmgpu/importer/legacy.py` - AST-based, lossless, snake_case importer with
  safe arithmetic eval and loud `import_notes`.
- `gates/gate_03.py` + `gates/legacy_canonical.py` + `gates/legacy_ground_truth.py`.
- `tests/test_importer.py` (per-file) + `tests/test_importer_basic.py`.
- `examples/handwritten_minimal.yaml` (PLAN 12.2 canonical doc).

## Gate result (real)
```
files: 50
schema-valid (in-process): 50/50
lossless vs canonical:    50/50
rmgpu validate (CLI):     50/50
rmgpu run minimal.yaml:   OK
JSON schema + hand example: OK
files with IMPORT-NOTES:  2
GATE STATUS: GREEN
```
Full pytest: **472 passed.**

## Key findings / decisions
- **Corpus = 50 files, not 47.** The step/job files say "47"; the real RMG-Py
  v4.0.0 input.py set is 50 (38 `examples/rmg/` + 12 `test/regression/`).
  Target treated as N==50 and met.
- **Lossless diff is non-circular.** The gate diffs the importer's resolved
  document against an *independent* AST canonicalizer
  (`gates/legacy_canonical.check_file`), not the importer's own dump
  (that would be circular and could not detect a dropped value).
- **Schema is lenient by design** (`extra="allow"`) so no legacy DSL value is
  ever silently dropped. Canonical strict usage = the hand-written example.
- **Two files carry documented last-wins IMPORT-NOTES** (repeated
  `simulator()`/`model()` calls in `minimal_staged` and `oxidation`): RMG
  replaces the block, the importer keeps only the last call and says so.
- No file failed to import; no value was dropped.

Per-step detail: `job-03-step-01-core.md`, `job-03-step-02-blocks.md`,
`job-03-step-03-cli.md`, `job-03-step-04-legacy.md`,
`job-03-step-05-gate.md`.

## For later jobs
- Consume the schema's typed blocks for known fields; `extra` keys exist only
  to preserve the legacy long-tail losslessly.
- `Species.structure` / `ForbiddenEntry.structure` coerce a bare SMILES string
  or dict to a `StructureValue`; the JSON schema accepts both forms.
- `rmgpu import <input.py> --to <out.yaml>` is the importer; notes are emitted
  as `# IMPORT-NOTE:` YAML comments.
