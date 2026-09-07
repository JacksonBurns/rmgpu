# job-07/step-06: The pdep driver + loop wiring + pdep/ output

Job: job-07 - Statmech + master equation (CSE) + pdep parity
Prereq: jobs 01-06 done (molecule, db, estimators, the core loop with its HPL pdep-stub, the reactor backend)
This step is part of that job. The job's overall goal:
Pressure dependence for unimolecular/reaction networks: statistical mechanics (DoS from vibrational/rotor/translation modes), the discretized master equation (CSE lumping - the default method), collision models, and the (T,P)-grid solving + Chebyshev/PDepArrhenius fitting that produces `Falloff` kinetics objects for the rate registry. NO QM anywhere (PLAN.md 8a): E0 from ML thermo, frequencies from the statmech DB, TS E0 derived from the HPL rate. The numerically hardest job in the project - CSE first, gated, before any other method (job 08).
The job's gate (run by the job's final step):
gates/gate_07.py (propane_branching, the R_Addition-heavy example, CSE in BOTH rmgpu and RMG-Py, same T/P grid): 1. Statmech unit parity: 50 species/intermediates from propane_branching: conformer assembly (E0, spin, mode counts) matches RMG-Py; Cp(T) on a 300-1500K grid within 1e-6 relative; DoS rho(E) on a grid within 1e-6 relative (spot-check 10, report max diff). 2. TS-E0 derivation: 20 pressure-dependent reactions: E0_TS (rmgpu, from the HPL rate) == E0_TS (RMG-Py) within 1e-8 (pins the no-QM path). 3. CSE k(T,P) parity (THE gate): for every pressure-dependent reaction: the fitted Falloff k(T,P) on the grid - max relative diff in k_inf, k_0, the broadening params + the Chebyshev/PDepArrhenius coefficients. TARGET: < 1% relative on k(T,P) values, < 1e-3 on coefficients. Any >1% reaction: listed + diagnosed (grain mismatch? collision? DoS?). 4. Network parity: network structure (isomers, channels, grain counts) matches RMG-Py (counts + a spot-check grain grid). 5. The full rmgpu run of propane_branching (pdep ON) completes; final core species/reaction counts vs RMG-Py (small divergence vs job 06's HPL run expected - the point is the k(T,P) parity above). A RED gate: port a second family's example (c3h4 has pdep reactions) to confirm it is not propane-specific.

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
2. STATUS.md: set your step row to `done` and append a session-log entry: `### <date> - job-07/step-06 / built: ... / checks: ... (GREEN|RED + one line) / commits: <hashes> / next: <the next step id>`. Update the top-level NEXT pointer to the next step's file.
3. Write the report to reports/job-07-step-06-driver.md with: what was built (files + ~1 line each), the checks run (the commands + the real results, not a paraphrase), the reference reads beyond the list (if any), the deviations from this file (if any, with the cause), and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.

## HANDOFF NOTES - session 1 (2026-09-07, context-exhausted; NO code built)

Progress: exploration only. The git tree was left CLEAN (no uncommitted
step-06 work; last commit b4eb540 = step-05 done). Nothing to salvage,
nothing half-written. Start from the plan below.

Verified anchors in /home/jackson/rmgpu/RMG-Py (read by line range, NOT whole
files - whole-file reads are what blew session 1's context):
- THE driver to port: `pressure_dependence(` at rmgpy/rmg/pdep.py:1376.
  Read its body first: family-metadata selection of pressure-dependent
  reactions, network building, (T,P) grid from the input's
  pressure_dependence block, the fit, Falloff attach.
- THE fit: `fit_interpolation_model(reaction, Tlist, Plist, K, model,
  Tmin, Tmax, Pmin, Pmax, error_check=False)` at
  rmgpy/pdep/reaction.pyx:338 - Chebyshev vs PDepArrhenius selected by the
  input's interpolation_model. Port this exact fit, do not substitute.
- rmgpy/exceptions.py:203 `class PressureDependenceError` - what RMG raises
  for unsupported methods (rmgpu: raise NotImplemented for MSC/RS/SLS).
- rmgpy/kinetics/chebyshev.pyx - read the FIT path (step 4's read may have
  covered only eval).
- rmgpy/pdep/network.py + rmgpy/pdep/collision.py - ALREADY ported in steps
  4-5; rmgpu/pdep/network.py and rmgpu/pdep/collision.py exist in-tree.

Open questions session 1 could NOT resolve - answer these FIRST (cheap
greps, a few minutes of tokens):
1. Where does the pressure_dependence block enter rmgpu? grep of
   rmgpu/importer/legacy.py for pressure_dependence|pressureDependence|
   interpolation found NO matches - the legacy importer does NOT carry it.
   Find the real path (core/loop.py? a model object? input.py DSL parse?)
   before finalizing run_pdep's signature.
2. Which legacy examples exercise pdep:
   `grep -rln "pressureDependence" /home/jackson/rmgpu/RMG-Py/examples/rmg/*/input.py /home/jackson/rmgpu/RMG-Py/test/regression/*/input.py`
   (session 1's run of this got lost to compaction; propane_branching is
   the gate example per the gate text above).
3. /home/jackson/rmgpu/rmgpu/examples/ appears absent/empty (ls returned
   nothing) - the 50 legacy input.py live under RMG-Py (38 examples/rmg +
   12 test/regression).

Useful pattern: scripts/record_job07_step04_reference.py (in-tree, ~11.4KB)
is step 04's reference recorder - reuse its shape if you need to record
RMG-Py reference data (network structure, grid, fitted coefficients) for
the step-07 gate.

Session 2 plan of attack:
  1. Answer the 3 open questions above (grep/sed only).
  2. Read pressure_dependence() body + fit_interpolation_model + the
     chebyshev fit path, by line range.
  3. Check rmgpu/kinetics/models.py for the Falloff/Chebyshev model shapes
     and rmgpu/core/loop.py for the HPL stub's exact call site (that is the
     wiring point for the loop deliverable).
  4. Write rmgpu/pdep/driver.py, do the loop wiring, add
     tests/test_pdep_driver.py.
  5. Run the Checks section until GREEN, then execute the Done protocol
     (exact) in full: commit, STATUS.md row + session-log entry + NEXT
     pointer, report at reports/job-07-step-06-driver.md (include these
     handoff notes' open questions as "reference reads beyond the list"
     context), then STOP.

## HANDOFF NOTES - session 2 (2026-09-07, context-exhausted mid-session; tree NOT clean)

Stopped per user directive. Everything below verified via git status/diff and
targeted reads at cutoff, so session 3 can trust it without re-deriving.

TREE STATE AT CUTOFF (verified):
- Last commit: f88c372 (the step-06 notes commit, no code).
- The ONLY uncommitted change: `rmgpu/pdep/network.py` (+16/-1). It adds a
  `state_provider` hook (a callable T -> per-T state dict) to
  `Network.__init__` and to the state-source dispatch in
  `calculate_rate_coefficients` (~line 600): the `state_at` dict (step-04
  path) takes precedence, else `state_provider`, else None. This is
  intentional step-06 work - KEEP IT and commit it as part of step-06
  (do NOT revert).
- `rmgpu/pdep/driver.py`: NOT created. `tests/test_pdep_driver.py`: NOT
  created. `rmgpu/core/loop.py`: untouched (loop wiring not done).
- `/tmp/probe_fit.py` exists (1.7KB): an empirical probe of RMG-Py's
  PDepArrhenius/Chebyshev fit unit conventions (fits known data, inspects
  stored coefficients vs CGS vs SI). Its run/output was NOT confirmed
  before cutoff - see FIRST ACTION below.

KEY FINDINGS (verified by read; the numerics need NO change for 2 isomers):
- The existing CSE machinery is already fully general over `n_isom`:
  `generate_full_me_matrix` / `apply_cse_allen` loop all isomers
  (rmgpu/pdep/network.py lines ~277-494). A 2-isomer test network needs no
  numeric changes. This step is orchestration/wiring only.
- `_apply_state` (rmgpu/pdep/network.py line 627) currently fills
  `dens_states` for a SINGLE-isomer state dict (product row at index
  `n_isom + n_reac`, line 639). It must be generalized so the driver's state
  provider can seed all `n_isom` rows (e.g. from `st["dens_isomers"]`), or
  the test builds per-isomer states - decide when writing the driver.
- UNIT-CONVENTION TRAP: rmgpu's `Chebyshev.__post_init__` (rmgpu/kinetics/
  models.py) has a hardcoded `-6` unit shift (likely the cm^3 -> m^3 shift on
  the A factor). The stored-coefficient convention (CGS vs SI) is THE open
  question for the fit port. The ported fit must produce coefficients in
  whatever convention rmgpu's existing Falloff/Chebyshev EVAL (job 02/3,
  rmgpu/kinetics/models.py) expects (SI). Verify via the probe + a line-range
  read of models.py BEFORE porting the fit. Do not guess the shift.

IN FLIGHT AT COMPACTION (resume exactly here):
1. FIRST ACTION: run the probe -
   `/home/jackson/miniforge3/envs/rmgpu/bin/python /tmp/probe_fit.py` - and
   read the output; it settles the CGS-vs-SI question. If /tmp was wiped,
   re-derive by reading, by line range ONLY: `fit_interpolation_model`
   (rmgpy/pdep/reaction.pyx:338) + the fit path in
   rmgpy/kinetics/chebyshev.pyx + PDepArrhenius in
   rmgpy/kinetics/arrhenius.pyx.
2. Re-answer the 3 open questions from the session-1 notes above (cheap
   greps). Q1 (where the pressure_dependence block enters rmgpu) decides
   `run_pdep`'s signature; the legacy importer does NOT carry it.
3. The loop wiring implies the loop/simulator evaluate path must carry `P`
   (k(T,P) at reactor conditions via the registry's evaluate(T,P)) - at
   cutoff I was about to confirm the simulator's `forward_A_T` call sites all
   take P; re-derive while reading core/loop.py.

SESSION 3 WORK ORDER (refined plan. BUDGET RULE: read by line range/sed,
NEVER whole files - whole-file reads are what killed sessions 1 and 2):
1. Probe run (above) + settle the unit convention; read the Falloff/
   Chebyshev/PDepArrhenius shapes in rmgpu/kinetics/models.py (line range).
2. Read by line range: the `pressure_dependence(` body at
   rmgpy/rmg/pdep.py:1376 (family-metadata selection, network building,
   (T,P) grid, fit, Falloff attach); `fit_interpolation_model` at
   rmgpy/pdep/reaction.pyx:338; the fit path in chebyshev.pyx.
3. Generalize `_apply_state` for multi-isomer (network.py line 627).
4. Write `rmgpu/pdep/driver.py` with `run_pdep(model,
   pressure_dependence_block, databases)`: family-metadata selection;
   network building (isomer grouping per RMG's rules); the (T,P) grid from
   the YAML block (Tmin/Tmax/Tcount, Pmin/Pmax/Pcount); per-(T,P) network
   solve; the EXACT RMG fit (port, no substitution); Falloff attach to the
   registry (job 02/3); write pdep/<network>.yaml (network def + k(T,P)
   grid, self-contained). MSC/RS/SLS: raise NotImplemented (job 08).
5. Loop wiring: replace the HPL stub in rmgpu/core/loop.py (find the exact
   call site - the pdep hook from job 06/2) so pdep families get Falloff
   kinetics; the loop's simulate step then uses k(T,P) at the reactor (T,P).
6. tests/test_pdep_driver.py: a hand-built 2-isomer network through
   run_pdep - check (a) Falloff attaches (the reaction's rate model is now a
   Falloff with the fitted params), (b) pdep/<network>.yaml writes + parses,
   (c) the fit reproduces the (T,P) grid within the interpolation tolerance
   (vs the network's direct solve).
7. Run the Checks section until GREEN, then the Done protocol (exact),
   lines 115-120 of this file, in full: commit (am, "job-07/step-06: ...",
   NO push), STATUS.md row -> done + session-log entry (exact format) + NEXT
   pointer to job-07/step-07-gate, report at
   reports/job-07-step-06-driver.md (include the session-1 open questions +
   this section as "reference reads beyond the list" context), then STOP.

Reference pattern: scripts/record_job07_step04_reference.py (step-04's
RMG-Py reference recorder, ~11.4KB). Steps 4-5 established the non-circular
pattern: record an RMG-Py run, then validate rmgpu against it. Reuse the
recorder's shape if you record a 2-isomer RMG-Py reference to validate the
driver (Falloff attach + yaml + fit-reproduces-grid).
