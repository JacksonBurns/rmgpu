# rmgpu -- RMG-GPU implementation repo

Ground-up, pure-Python (NumPy/PyTorch) rewrite of RMG. This repo is the working
repository: the plan, the orchestration docs, the job/step prompts that drive
implementation across sessions, and (from job 00 on) the code itself.

## Layout

    PLAN.md            The full feasibility + feature-parity plan (read the sections
                       your step file names; it is the source of truth for design).
    ORIENTATION.md     Self-contained context brief. Read ONCE at the start of a
                       job (see "Session vs step" below), never per step.
    STATUS.md          Work tracker: NEXT pointer, job/step tables, decisions
                       log, append-only session log.
    prompts/           One file per JOB (a job is a milestone with one gate):
                       job-NN-<name>.md -- short brief + step list only.
    prompts/steps/     One file per STEP (a step is one subagent session):
                       job-NN-step-MM-<name>.md -- self-contained, budgeted.
    rmgpu/             (created in job 00) the package itself
    gates/             (created in job 00) parity-gate scripts + baselines
    reports/           (created in job 00) gate/step reports written by steps

Reference material (read-only, NOT part of this repo, at /home/jackson/rmgpu):
  - RMG-Py/          reference implementation (v4.0.0-5-gd08392ed), conda env
                     rmg_env has it installed
  - RMG-database/    the 2.4M-line data (families, libraries, statmech,
                     transport), access via RMG-Py and its rmg_env
  - rmgdb/           SQL wrapper over RMG-database (our database layer),
                     installed in the rmgdb conda environment
  - chemprop_example/  THE reference for Chemprop inference. predicting.ipynb
                     shows exactly how to load a trained Chemprop v2 checkpoint
                     (models.MPNN.load_from_checkpoint) and predict (featurize
                     with featurizers, build a MoleculeDataset,
                     data.build_dataloader, pl.Trainer(...).predict). Use the
                     chemprop-dev environment if needed. See below: "The ML
                     estimators are NEW".

## The ML estimators are NEW (read this)

RMG-Py ships a Chemprop-based thermo estimator (`rmgpy/ml/estimator.py`, an
`MLEstimator` wrapping Chemprop species models; disabled on py3.11, upstream
issue #2559). In rmgpu it is REPLACED, not reused:

  - rmgpu's estimators (`rmgpu/ml/thermo_estimator.py`,
    `rmgpu/ml/kinetics_estimator.py`) are written fresh, modeled on
    chemprop_example/predicting.ipynb. RMG's `ml/estimator.py` is read ONLY to
    learn its checkpoint layout (Hf298 model + S298+Cp model), its uncertainty
    cutoffs, and how the `ml_estimator:` input DSL wires them - as design
    reference. Nothing is imported, copied, or wrapped from it.
  - Same for checkpoints: the existing checkpoints (CheMeleon thermo models,
    any reaction-model checkpoints) are CONSUMED via the new estimators'
    own load path - which mirrors the chemprop_example load path
    (MPNN.load_from_checkpoint + the model's featurizer). Do not try to make
    RMG's MLEstimator work.
  - If a new/better checkpoint arrives later, only the `ml_estimator:` block of
    the input YAML changes (checkpoint name + hash). No code change. (PLAN.md
    8a.3: the checkpoint interface is the only seam between model improvement
    and mechanism generation; model improvement happens OUTSIDE this package.)

## How work gets done (read this before starting)

Two levels, both driven by a parent (coordinator) agent:

  JOB   = a milestone with one final gate. Too big for one session - that is
          why this framework exists. A job brief (prompts/job-NN-*.md) is a
          short file: goal, step list, the job's gate definition.
  STEP  = one self-contained unit of work, sized to fit a SINGLE fresh
          subagent session (target: 1 step per session; a session may finish
          2 small steps if its context budget allows, but never start step 3
          after doing 2). Each step file (prompts/steps/job-NN-step-MM-*.md)
          names exactly which reference files to read, what to build, the
          step's checks, and its context budget.

### Session vs step (the loop)

The human starts the "coordinator": a fresh agent session.

The coordinator:
  0. Reads the contents of this README.md
  1. Reads STATUS.md -> the NEXT pointer names exactly one step file.
     (It does NOT read ORIENTATION.md or the job brief per step - those are
     read once, when the step is the first of its job, and the step file
     repeats the few facts it needs.)
  2. Spawns ONE subagent for that step. The subagent's task prompt is
     essentially: "Read prompts/steps/<file> and do it. Everything you need
     is in that file plus what it points at."
  3. Spawn ONE subagent to: review the step's commit and report results -- it should run the step's
     checks if the report looks off --- the coordinator then update STATUS.md's
     NEXT pointer to the following step.
  4. If a step's gate/job-gate is RED, or a step comes back incomplete,
     the NEXT pointer targets a fix step (write a small one in
     prompts/steps/ if needed, e.g. job-04-step-11-fix-*.md) before moving
     on.
  5. Repeat. A step's self-report is not definitive proof; the checks are.
     Trust the small details, spot-check the major claims.

Rules that make this work (why the structure is this way):

- A single step reads at most ~3k lines of reference code (each step file
  states its budget). That keeps any one session under compaction pressure;
  the job's 50-150k lines of reference are spread across 4-9 steps, each of
  which re-reads only its own slice.
- Steps within a job are strictly sequential and share one branch; a step's
  output files are the next step's input. Nothing is parallelizable - one
  subagent at a time, because every heavy task on this machine uses the same
  GPU. Subagents MUST NOT spawn subagents (tell every subagent this).
- Do not start job N's steps until job N-1's gate is GREEN and recorded in
  STATUS.md. Exceptions require a decisions-log entry.
- Gates: a job's final step runs the job gate (a script in gates/, report in
  reports/). A step's intermediate checks are lighter (unit tests, small
  scripts); a step's report goes to reports/job-NN-step-MM.md. Never
  fabricate or "smooth" a result; a red check is a finding, recorded as such.
- Commits: one or more per step, message prefix "job-NN/step-MM: <summary>".
  No pushes.
- Conda env: `rmgpu` (created in job 00). NEVER install into system Python
  or the active venv. Always invoke the interpreter directly, e.g.
  /home/jackson/miniforge3/envs/rmgpu/bin/python ...
- The coordinator does NO implementation work itself. It orchestrates: pick
  step, spawn subagent, spawn review agent, updates STATUS.md, commit STATUS.md changes.
  **ALL code work happens in subagents** (this is **PIVOTAL** - rely on subagents, be
  protective of your context window).

## Roadmap (see PLAN.md section 10 for the full version)

    00  env + package skeleton + test scaffolding
    01  units + molecule layer (RDKit wrapper, adjlist, atom types, resonance)
    02  database layer via rmgdb + round-trip hash gate
    03  YAML input schema + CLI + legacy .py importer
    04  ML estimators (CheMeleon thermo, Chemprop kinetics) + rate registry
        (THE thesis test; estimators built fresh per chemprop_example)
    05  reaction recipe DSL + product enumeration (port from family.py)
    06  core/edge loop + torchdae reactor (first integration; superminimal/c3h4)
    07  statmech + master equation (CSE) + pdep gate (propane_branching)
    08  pdep MSC/RS/SLS + interpolation + isotope + observables/diff/merge
    09  sensitivity/uncertainty via torchdae adjoint
    10  full gas-phase parity (regression suite)  <-- the parity milestone
    11  plugin protocol + solvation plugin + liquid reactors
    12  catalysis plugin (post-parity; not required for core parity)
