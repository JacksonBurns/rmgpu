# job-06/step-05: Chemkin writer + species dictionary

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

- Previous step (already done, its code is in the tree): job-06/step-04-output
- This step: job-06/step-05-chemkin
- Next step (do NOT start it): job-06/step-06-gate

## Goal

The legacy-format writer: chemkin.inp + chem_annotated.inp +
species_dictionary.txt from the canonical artifact (the WRITE path -
reading lands in job 08). Interop is table stakes.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/rmg/chemkin.pyx  (the WRITE path ONLY:
    the chem.inp/chem_annotated.inp layout - phases, species (formula +
    NASA/Wilhoit coeffs), reactions (rate model lines: A n Ea, the
    falloff lines, the third-body lines), the species_dictionary.txt
    (adjacency lists); the EXACT format - RMG's output is the spec)
  RMG-Py/examples/rmg/*/output (a real RMG-Py chem.inp + species_dictionary
    .txt from an example run, if present - else generate one with a small
    RMG-Py run in env rmg_env: the format reference)
  (job-02's entries; job-06/5's mechanism schema)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/io/chemkin.py:
    `write_chemkin(mechanism, path)` -> chem.inp + chem_annotated.inp +
    species_dictionary.txt, byte-format-compatible with RMG-Py's writer
    (the format: the phase lines, the species thermo lines (NASA7 or
    Wilhoit - RMG writes Wilhoit as a NASA-equivalent where needed -
    check RMG's actual output and match it), the reaction rate lines
    (including falloff + third-body lines), the comments (annotated has
    them)).
    From the canonical artifact (the schema's export) - not from the live
    model (the writer works on the serialized mechanism).
- tests/test_chemkin.py: a hand-built mechanism (2 species, 2 reactions:
  one Arrhenius, one falloff) -> chem.inp: parse it back with a minimal
  checker (species count, reaction count, the rate params round-trip);
  against a REAL RMG-Py output (from the reference run): the format
  matches line-for-line for the same mechanism (the reference mechanism is
  generated by RMG-Py itself - a small script).

## Checks (must run and pass before you claim done)

  pytest tests/test_chemkin.py -q -> all pass
  the format check vs RMG-Py's own chem.inp (line-structure diff; the
    mechanism content matches by construction)

## Pitfalls

- The Chemkin format is fussy (the annotated vs plain variants,
  the falloff line syntax, the third-body list) - match RMG-Py's ACTUAL
  output, not the textbook format; RMG's writer is the spec.
- Wilhoit -> chemkin: RMG converts Wilhoit to a NASA7-like representation
  for chem.inp where needed - check what RMG actually writes and match it
  (note the decision in the report).

## Done protocol (exact)

1. Commit the code: `git commit -am "job-06/step-05: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-06/step-05 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-06-step-05-chemkin.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
