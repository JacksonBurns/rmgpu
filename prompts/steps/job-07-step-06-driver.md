# job-07/step-06: The pdep driver + loop wiring + pdep/ output

Job: job-07 - Statmech + master equation (CSE) + pdep parity
Prereq: jobs 01-06 done (molecule, db, estimators, the core loop with its HPL pdep-stub, the reactor backend)
This step is part of that job. The job's overall goal:
Pressure dependence for unimolecular/reaction networks: statistical mechanics (DoS from vibrational/rotor/translation modes), the discretized master equation (CSE lumping - the default method), collision models, and the (T,P)-grid solving + Chebyshev/PDepArrhenius fitting that produces `Falloff` kinetics objects for the rate registry. NO QM anywhere (PLAN.md 8a): E0 from ML thermo, frequencies from the statmech DB, TS E0 derived from the HPL rate. The numerically hardest job in the project - CSE first, gated, before any other method (job 08).
The job's gate (run by the job's final step):
gates/gate_07.py (propane_branching, the R_Addition-heavy example, CSE in BOTH rmgpu and RMG-Py, same T/P grid): 1. Statmech unit parity: 50 species/intermediates from propane_branching: conformer assembly (E0, spin, mode counts) matches RMG-Py; Cp(T) on a 300-1500K grid within 1e-6 relative; DoS rho(E) on a grid within 1e-6 relative (spot-check 10, report max diff). 2. TS-E0 derivation: 20 pressure-dependent reactions: E0_TS (rmgpu, from the HPL rate) == E0_TS (RMG-Py) within 1e-8 (pins the no-QM path). 3. CSE k(T,P) parity (THE gate): for every pressure-dependent reaction: the fitted Falloff k(T,P) on the grid - max relative diff in k_inf, k_0, the broadening params + the Chebyshev/PDepArrhenius coefficients. TARGET: < 1% relative on k(T,P) values, < 1e-3 on coefficients. Any >1% reaction: listed + diagnosed (grain mismatch? collision? DoS?). 4. Network parity: network structure (isomers, channels, grain counts) matches RMG-Py (counts + a spot-check grain grid). 5. The full rmgpu run of propane_branching (pdep ON) completes; final core species/reaction counts vs RMG-Py (small divergence vs job 06's HPL run expected - the point is the k(T,P) parity above). A RED gate: port a second family's example (c3h4 has pdep reactions) to confirm it is not propane-specific.

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

- Previous step (already done, its code is in the tree): job-07/step-05-collision
- This step: job-07/step-06-driver
- Next step (do NOT start it): job-07/step-07-gate

## Goal

The pdep driver: the orchestration that selects pressure-dependent
reactions, builds networks, runs the (T,P) grid, fits to
Chebyshev/PDepArrhenius, attaches Falloff to the reactions, and wires it
into the core loop (replacing job 06's HPL stub) + writes pdep/<network>.
yaml.

## Reference to read (this step's budget)

  RMG-Py/rmgpy/rmg/pdep.py  (1000 - the PDEPENDENCE DRIVER: how
    RMG decides which reactions get pdep (the family metadata), builds
    networks (grouping isomers per RMG's network rules), the (T,P) grid
    from the input's pressure_dependence block (Tmin/Tmax/Tcount,
    Pmin/Pmax/Pcount), fitting to Chebyshev/PDepArrhenius, the
    interpolation_model selection - port this orchestration; it connects
    job 06's pdep stub to the engine)
  (the network + collision from steps 4-5; the registry's Falloff/Chebyshev
    from job 02/3; the core loop's pdep hook from job 06/2)
  RMG-Py/rmgpy/kinetics/chebyshev.pyx (re-read the FIT if step 4's read
    covered only the eval - the fit RMG uses)

Read ONLY what is listed plus the direct dependencies you hit (note any extra reads in the report). The budget is sized so the listed reads + the deliverables fit ONE session without context compaction - if you find the reads are bigger than that, STOP and record it in the report (a step whose reads overflow is a framework bug, not something to push through).

## Deliverables

- rmgpu/pdep/driver.py:
    `run_pdep(model, pressure_dependence_block, databases)`:
      select the pressure-dependent reactions (from the family metadata -
      the families flagged pdep in the DB);
      build the networks (group isomers per RMG's network rules - the
      same isomers share a network);
      run the (T,P) grid (from the YAML pressure_dependence block:
      Tmin/Tmax/Tcount, Pmin/Pmax/Pcount) - for each (T,P): solve the
      network (step 4) -> the k(T,P) for every reaction in it;
      fit k(T,P) to Chebyshev/PDepArrhenius (the interpolation_model from
      the YAML; port RMG's exact fit - do not substitute a different fit);
      attach the Falloff (the fitted params) to the reactions' rate models
      (the registry, job 02/3);
      write pdep/<network>.yaml (PLAN.md 12.3: the network definition +
      the k(T,P) grid - self-contained).
    The (T,P) grid can be large (Tcount x Pcount x reactions x networks) -
    batch the solves on GPU where it helps (torch) - but SEQUENTIAL across
    the shared GPU (no llama-server contention); record wall-time (the
    "leverage GPUs" payoff - show it in the report).
- The loop wiring (job 06/2's hook): replace the HPL stub - pressure-
  dependent families now get Falloff kinetics (the driver runs at the point
  in the loop RMG runs it - after the reactions are estimated, before the
  simulate/screen that consumes k(T,P) at the reactor conditions). The
  loop's simulate step then uses k(T,P) (the registry's evaluate(T,P) -
  the Falloff model) at the reactor's (T,P).
- The driver's method selection: the YAML's pressure_dependence.method
  (normalized per step 4's table) - CSE works; MSC/RS/SLS raise
  NotImplemented (job 08).
- tests/test_pdep_driver.py: a small pdep system (a hand-built 2-isomer
  network) through the driver: the Falloff attaches (the reaction's rate
  model is now a Falloff with the fitted params), the pdep/<network>.yaml
  writes + parses, the (T,P) grid solve produces the expected k(T,P)
  (vs the network's direct solve - the fit's error is within the
  interpolation tolerance).

## Checks (must run and pass before you claim done)

  pytest tests/test_pdep_driver.py -q -> all pass
  the small pdep system: Falloff attached + pdep/<network>.yaml valid +
    the fit reproduces the grid within the interpolation tolerance

## Pitfalls

- The driver is ORCHESTRATION - the numerics are steps 1-5; a bug
  here is a wiring bug (wrong reaction selected, wrong grid, wrong fit
  model), not a numerics bug. The gate's k(T,P) parity (step 7) catches
  the numerics; the driver's own test (this step) catches the wiring.
- The (T,P) grid batching: the GPU payoff - batch the (T,P) solves
  (torch) where the network allows; but the single-GPU-task rule holds
  (sequential with llama-server). Record the wall-time (the plan wants the
  "leverage GPUs" number shown).
- Do NOT port MSC/RS/SLS here - the driver raises NotImplemented for them
  (job 08 lands them). The gate (step 7) is CSE-only.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-07/step-06: <summary>"` (several commits are fine; NO push).
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-07/step-06 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Do NOT set the top-level NEXT pointer - the coordinator does.
3. Write the report to reports/job-07-step-06-driver.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
