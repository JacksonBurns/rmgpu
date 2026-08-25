# job-06/step-01: Reactor definitions + termination + torchdae backend

Job: job-06 - Core/edge mechanism loop + torchdae reactor (first integration)
Prereq: jobs 01-05 done (all of: molecule, db, schema, estimators, recipes)
This step is part of that job. The job's overall goal:
The `rmgpu run` command actually works end to end (gas phase, constant T or T/P): load YAML -> build model from seed species -> estimate properties (libraries/ML) -> enumerate candidate reactions (recipes) -> simulate in torchdae -> screen by conversion -> grow/prune -> iterate to steady state -> write the output tree (PLAN.md 12.3), including the canonical `mechanism/core.yaml`. The moment of truth for the stack.
The job's gate (run by the job's final step):
gates/gate_06.py: 1. torchdae sub-gate: a stiff reference ODE (a 3-reaction Lindemann falloff toy system, or Van der Pol mu=10 as a non-chemistry control) integrated by torchdae vs a high-accuracy reference (RK45 tiny step) - max abs diff recorded (PLAN.md 13 risk 3: prove it before trusting it with mechanism growth). 2. `rmgpu run` on the imported `superminimal` (HPL kinetics, pdep off/stub): completes to steady state (or max_iter); iteration count + core/edge species+reaction counts recorded. 3. Parity vs RMG-Py (pdep OFF, same input; its output -> gates/baselines/superminimal/): |core_rmgpu - core_rmg| and |core+edge_rmgpu - (core+edge)_rmg| as sets. TARGET: core identical (or documented divergence with cause); edge within a small fraction. Any systematic divergence (a family always missing/extra): fix if tractable in this job, else record precisely. 4. Same for `c3h4` (the right size - NOT the GRI-scale example). 5. Output tree: every file in PLAN.md 12.3 exists and is valid (YAML parses, CSV columns right, core.yaml loads back via the schema and re-simulates the final iteration's profiles within tolerance). 6. provenance.yaml contains real hashes/versions (not placeholders).

First step of this job: skim /home/jackson/rmgpu/rmgpu/ORIENTATION.md once
(what stays/goes/external; the master-equation data path; conventions). Later
steps of this job do not need it - everything they need is in their own file.

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

- Previous step (already done, its code is in the tree): (this is the first step of the job)
- This step: job-06/step-01-reactor
- Next step (do NOT start it): job-06/step-02-model

## Goal

The reactor layer: the reactor dataclasses (simple, const_V,
const_TP + the energy balance), termination criteria, and the torchdae
backend (the SOLE solver) with the stiff-ODE validation sub-gate.
No mechanism loop yet - just: a set of reactions + a reactor -> profiles.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/solver/base.pyx  (1402 - the reactor base: mole
    balance dN/dt = nu * r, the initial-concentration setup, the event/
    termination plumbing - port the MATH)
  RMG-Py/rmgpy/solver/simple.pyx  (1061 - SimpleReactor (isothermal batch) +
    ConstantVReactor + ConstantTPReactor (the energy balance ODE); the
    termination conditions (conversion, time, criticality))
  RMG-Py/rmgpy/solver/termination.py  (66 - the termination conditions)
  torchdae (installed; read its API: the integrator call signature, event
    syntax - 0.1.1; if the API differs from PLAN.md 5's description, adapt
    + document)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/reactor/reactors.py: reactor definitions (dataclasses):
    SimpleReactor (isothermal batch), ConstantVReactor,
    ConstantTPReactor (with the energy balance ODE - ported from
    simple.pyx/base.pyx), + stub types (Liquid/MBSampled -> job 11, Surface
    -> job 12 - raise NotImplemented with the job reference).
    Termination criteria (conversion, time, criticality) ported exactly
    (they drive the screen in job 06/3's loop).
- rmgpu/reactor/torch.py: the torchdae backend (SOLE solver):
    build the ODE right-hand side as a torch function: state = moles (or
    mole fractions + total), y' = nu @ rates(y, T, P), rates from the rate
    registry (job 02/04 - evaluate(T,P) per reaction); transport NOT needed
    for gas (ideal).
    integrate with torchdae (BDF2 or TR-BDF2; record the choice) on cuda
    if available else cpu (single GPU task rule).
    events/termination: torchdae events for conversion/time termination;
    observables sampled on a time grid.
    API: simulate(mechanism, reactor, T, P, t_end, termination) ->
    Profiles (times, species amounts + molefractions, termination info).
    SI units. Mole fractions by default in the output (the 2.3.0 "moles vs
    fractions" footgun is fixed - PLAN.md 12.3).
- VALIDATION SUB-GATE (before anything else trusts this): integrate a stiff
  reference ODE (a 3-reaction Lindemann falloff toy system built from the
  rate registry, or Van der Pol mu=10 as a non-chemistry control) with
  torchdae vs a high-accuracy reference (scipy RK45 tiny step - reference
  only, NOT a backend); max abs diff recorded.
- tests/test_reactor_torch.py: the stiff-ODE sub-gate as a test; mole
  balance closes (sum of mole fractions == 1 within solver tol); a
  2-reaction A<->B toy system reaches the expected equilibrium; conversion
  termination fires at the right time; the energy balance (const_TP) keeps
  T constant (the ODE enforces it).

## Checks (must run and pass before you claim done)

  pytest tests/test_reactor_torch.py -q -> all pass
  the stiff-ODE sub-gate: max abs diff vs RK45-tiny-step recorded (< the
    documented tolerance)

## Pitfalls

- The ODE right-hand side is a TORCH FUNCTION - job 09's adjoint
  depends on it being differentiable through the rates; keep it a pure
  torch computation (numpy fallback is fine inside, but the graph must
  exist - document how rates flow: if the registry's numpy path breaks the
  graph, the rates must be computed in torch for the sensitivity path;
  note it).
- torchdae 0.1.1 is young (PLAN.md 13 risk 3): if the API fights you
  (events, DAE vs ODE), record the exact limitation - the sub-gate exists
  so we know now, not at job 10.
- Single GPU task: the box shares one GPU with llama-server - sequential
  solves, batch within a solve, small memory footprints.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-06/step-01: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-06/step-01 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-06-step-01-reactor.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
