# job-11: Plugin protocol + solvation plugin + liquid reactors

Read ORIENTATION.md and PLAN.md section 9 (plugin design - 9.1 protocol, 9.3
solvation) in full. Prereqs: job-10 (gas-phase parity green). This job validates
the plugin protocol with the LIGHT plugin (solvation) before the heavy one
(catalysis, job 12), and lands the liquid-phase reactors.

## Goal

1. The `RmgpuPlugin` protocol from PLAN.md 9.1, wired into the core loop (the core
   stays agnostic to which plugins are loaded).
2. The SOLVATION plugin: solvated thermo/kinetics providers + the LiquidReactor /
   MBSampledReactor, so `rmgpu run` on a liquid-phase example works.
3. A working liquid-phase parity example.

## Reference (read)

  PLAN.md 9.1 - the exact protocol (register_species/families/thermo/kinetics/
    reactor + on_new_species/on_new_reaction/after_prune + load_database).
  RMG-Py/rmgpy/data/solvation.py (111k) - implicit-solvation (SMD-like)
    corrections parameterized in the solvation DB; the solvated-thermo and
    rate-correction logic.
  RMG-Py/rmgpy/solver/liquid.pyx (34k) - the LiquidReactor (Nernst, liquid-phase
    mole balances, the liquid mass-transfer coefficient power law from the input
    DSL), mbSampled.pyx (25k) - the MB-sampled reactor.
  RMG-Py/rmgpy/data/solvation.py + input/solvation/ (RMG-database) - the solvation
    groups/libraries; rmgdb solvation.db has the tables (job-02 SolvationDB).
  Examples: examples/rmg/liquid_phase, liquid_phase_constSPC, liquid_cat (cat is
  job 12 - skip it), the RMS_CSTR_liquid_oxidation + liquid_oxidation regression
  inputs.

## Deliverables

1. `rmgpu/plugins/base.py` - the `RmgpuPlugin` ABC per PLAN.md 9.1 + the plugin
   loader (entry-point discovery: plugins register via `rmgpu.plugins` entry point
   group; core calls the hooks in fixed order, merges results). The core loop
   (job-06 model.py) gets the hook call-sites (on_new_species, on_new_reaction,
   after_prune, the register_* at init) - keep them small and unconditional.
2. `rmgpu/plugins/solvation/` - the solvation plugin:
   - `register_thermo`: a thermo provider that, when the run is in solution
     (YAML `solvation:` block present), returns gas_thermo (ML/library) +
     solvation free-energy correction (from solvation.db via the SolvationDB
     facade). Port RMG's SMD-correction semantics from data/solvation.py.
   - `register_kinetics`: the solvation correction to rates (reaction
     free-energy-of-solvation shifts the barrier) - port from data/solvation.py.
   - `register_reactor`: LiquidReactor + MBSampledReactor (port liquid.pyx /
     mbSampled.pyx math onto the torchdae backend).
   - `load_database`: pull the solvation groups/libraries (rmgdb solvation.db).
   - `on_new_species`: enforce the liquid-phase species constraints (which
     species may be in the liquid - RMG's liquid-phase screening).
3. Wire the YAML `solvation:` block (job-03 schema) to enable the plugin:
   when present, the run is liquid-phase (reactors constrained to liquid types,
   solvation providers active).
4. `reports/job-11.md` documents the plugin protocol as-implemented (any
   deviation from PLAN.md 9.1 - there should be none; if there is, explain why
   and update PLAN.md).

## Gate (job 11) -> gates/gate_11.py, report reports/job-11.md

  1. Protocol test: a trivial no-op plugin (register_* no-ops) loads and a
     gas-phase run (superminimal) is byte-identical to the no-plugin run
     (proves the hooks are inert when unimplemented - the core is agnostic).
  2. Solvation thermo parity: for 25 solvated species: rmgpu solvated thermo ==
     RMG-Py solvated thermo (gas value + SMD correction), tolerance 1e-8
     (generate RMG-Py reference). The correction terms themselves (delta G solv)
     match, not just the sum (catches sign/unit bugs).
  3. Liquid reactor parity: run the `liquid_phase` example in rmgpu vs RMG-Py:
     final mechanism sets (core species/reaction Δ, same as job 10) + observable
     (conversion vs time) max diff. `liquid_phase_constSPC` too if time permits.
     The liquid_oxidation regression input: mechanism-set parity.
  4. MB-sampled: run an MB-sampled example (find one in test/regression - the
     RMS_ ones) - reactor profiles match RMG-Py within solver tolerance.
  5. Plugin isolation: the solvation plugin's code touches ONLY rmgpu/plugins/
     + the core hook call-sites (verify by diff: no solvation code in rmgpu/core,
     rmgpu/ml, rmgpu/reactor).

## When done

STATUS.md + commit "job-11: plugin protocol + solvation plugin + liquid reactors"
+ STOP.

## Pitfalls

- The plugin protocol is the deliverable - get the hook semantics EXACTLY as
  PLAN.md 9.1 (they were designed so solvation = 2 providers + reactors,
  catalysis = all of them). If you change the protocol, PLAN.md 9 must change with
  it (the catalysis job builds on it).
- Liquid-phase thermodynamics is different (activities, not partial pressures;
  the liquid mole-fraction basis). Port RMG's liquid basis handling exactly -
  the mole-fraction-vs-activity confusion is the classic bug here.
- The MB-sampled reactor is a Monte-Carlo sampling of the rate - port RMG's exact
  sampling (it is not a standard solver; it is RMG's own method).
