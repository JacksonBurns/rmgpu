# job-10: Full gas-phase feature parity (the parity milestone)

Read ORIENTATION.md and PLAN.md section 10 (Phase 4: "full core parity"). Prereqs:
jobs 01-09 done and their gates green (esp. 04 thesis-test, 06, 07). This job
does NOT add features - it runs the whole regression battery and drives the
remaining gas-phase examples to green. It is a campaign job; expect it to span
multiple sessions.

## Goal

`rmgpu run` reproduces RMG-Py's gas-phase behavior across the standard example
+ regression battery, with parity measured the way PLAN.md 12.5 prescribes:
diff the canonical `mechanism/core.yaml` artifacts (NOT the Chemkin round-trip),
plus observable-level comparison via job-08's observables.

## Scope (the gas-phase battery)

From the 47 legacy inputs, the gas-phase set (inventory in job 03) minus:
  - liquid/liquid_cat (job 11), surface/multisurf (job 12),
  - the MB-sampled ones (job 11).
Prioritize (this order; each is a sub-deliverable with its own gate line):
  1. superminimal, c3h4 (re-confirm job-06 numbers with pdep ON)
  2. minimal, minimal_thermofilter, minimal_staged, minimal_sensitivity (job 09),
     pruning_test, heptane-filterReactions (the filter/prune logic)
  3. ethane-oxidation, propane_branching (pdep), ch3no2, nox_transitory_edge,
     minimal_dynamics
  4. c3h4-scale: heptane-eg5, e85, diesel, aromatics, nitrogen, oxidation
     (the big regression suite - these are the expensive ones; run them, but
     expect the longest sessions; parallelize across SEPARATE sessions if needed,
     never two GPU tasks at once)
  5. The fragment/seed-mechanism examples (external_library, gri_mech_rxn_lib,
     g3/mr/sr_test, TEOS) - these exercise library override + seeding.

## Per-example protocol (repeat; it is the unit of work)

  1. Import the example's input.py to YAML (job-03 importer). If it was a
     documented IMPORT-NOTE case, resolve the note now (the example must run).
  2. Run RMG-Py on the original input.py (same machine, same DB). Capture its
     output (final mechanism + profiles) to gates/baselines/<example>/.
  3. Run rmgpu on the imported YAML. Capture the output tree.
  4. Parity report (reports/parity/<example>.md):
     - core species set: |Δ| (exact match target)
     - core reaction set: |Δ| (exact match target)
     - edge: |Δ| (small fraction acceptable; list stragglers)
     - observables (job-08): max abs/rel diff per observable type
     - any systematic divergence: root-cause it (family missing? estimator
       difference? pdep fit? screening?) and record.
  5. Flip the example status in the parity matrix below.

## The parity matrix (maintain in reports/parity/INDEX.md; one row per example)

| example | core species Δ | core rxn Δ | edge Δ | observables | status | notes |
|---------|---------------|-----------|--------|-------------|--------|-------|

Status: pending | pass | pass-with-deviation (documented) | fail (blocked, cause recorded)

## Gate (job 10) -> gates/gate_10.py, report reports/job-10.md

  - gate_10.py reads reports/parity/INDEX.md + the per-example reports and
    asserts: all gas-phase battery examples are `pass` or `pass-with-deviation`;
    `fail` count is 0 (or each `fail` has an accepted, documented cause in
    STATUS.md decisions log - user-approved).
  - reports/job-10.md: the full INDEX.md inline + a summary: N pass, N
    pass-with-deviation (list + one-line cause each), N fail (list + cause).
  - This is THE parity milestone (PLAN.md 10 Phase 4). A green gate means
    "rmgpu reproduces RMG's gas-phase mechanism generation" - the headline
    result of the PoC. Be rigorous; do not cherry-pick examples.

## When done (or when the battery is exhausted for a session)

STATUS.md: update the parity matrix reference + a session-log entry listing which
examples advanced this session + commit hashes. Commit "job-10: parity <examples>
advanced". If the full battery is green, flip the job-10 row to done and STOP the
job. Otherwise STOP and report which examples remain.

## Pitfalls

- Divergences usually cluster by CAUSE (one bad family matcher, one estimator
  edge case, one pdep fit), not by example. When you find a systematic cause,
  fix it in the responsible module (job 04/05/06/07) - not by special-casing the
  example - and re-run the affected examples.
- The big examples (diesel, heptane-eg5) are slow; use the torchdae GPU batch and
  the core-loop's max-edge-species cap from the input. Do not lower the cap to
  force a match - parity means same input, same parameters.
- Do NOT "tune" thresholds (toleranceKeepInEdge etc.) to force agreement - the
  imported YAML carries RMG's exact values.
