# STATUS - rmgpu work tracker

Single source of truth for work state across sessions. Every session updates this
file before committing. Do not delete entries; append and annotate.

## Job table

| Job | Title | Gate | Status | Commit(s) |
|-----|-------|------|--------|-----------|
| 00 | Env + package skeleton + test scaffolding | smoke test | pending | - |
| 01 | Units + molecule layer | adjlist round-trip + isomorphism vs RDKit | pending | - |
| 02 | Database layer via rmgdb + round-trip | entry-count + table hash vs RMG-Py | pending | - |
| 03 | YAML input schema + CLI + legacy importer | 47 example input.py -> yaml, lossless | pending | - |
| 04 | ML estimators + rate registry | Hf298/S298/Cp, HPL k(T) vs RMG-Py (coverage+error report) | pending | - |
| 05 | Reaction recipe DSL + product enumeration | product sets vs RMG-Py on training families | pending | - |
| 06 | Core/edge loop + torchdae reactor | superminimal + c3h4 core/edge vs RMG-Py | pending | - |
| 07 | Statmech + master equation (CSE) + pdep | k(T,P) falloff vs RMG-Py (propane_branching) | pending | - |
| 08 | pdep MSC/RS/SLS + interpolation + observables/diff/merge | pdep method diffs + observables regression | pending | - |
| 09 | Sensitivity/uncertainty via torchdae adjoint | sensitivity vs RMG-Py finite-difference | pending | - |
| 10 | Full gas-phase parity (regression suite) | checkModels-style suite GREEN | pending | - |
| 11 | Plugin protocol + solvation + liquid reactors | liquid_phase example vs RMG-Py | pending | - |
| 12 | Catalysis plugin (post-parity) | minimal_surface example | pending | - |

Status values: pending | in-progress | done | blocked
A job is "done" only when its gate is GREEN (or its gap is an explicitly-accepted,
documented finding) and the session log has the evidence.

## Decisions log (append, do not edit)

- [plan] QM out of scope entirely (PLAN.md 8a.3). Model improvement happens outside
  the package; checkpoint interface is the seam.
- [plan] No fallbacks of any kind (PLAN.md 1, 14). ML is the only estimator; torchdae
  is the only reactor backend.
- [plan] Catalysis + solvation are plugins, built only after core parity (PLAN.md 9).
  Solvation first (validates protocol), catalysis second.

## Session log (append newest at bottom)

(append entries below this line; format:
  ### <date> - job-NN
  built: ...
  gate: GREEN|RED - <one-line evidence>
  commits: <hashes>
  next: <what the next session should do first>)
