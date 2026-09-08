# job-07/step-06 report: The pdep driver + loop wiring + pdep/ output

## What was built

Files (new unless noted):

- **`rmgpu/pdep/driver.py`** (new, ~540 lines) - the pdep DRIVER, the
  orchestration between the core loop and the master-equation numerics
  (steps 01-05). `run_pdep(network, block, method, interpolation_model,
  Tlist, Plist, output_dir, error_check)`: (1) method selection (CSE works;
  MSC/RS/SLS raise NotImplementedError for job 08; unknown -> PDepDriverError),
  (2) the (T,P) grid from the YAML pressure_dependence block via RMG's
  exact Gauss-Chebyshev (Chebyshev model) / linear-log (PDepArrhenius)
  generate_T_list/generate_P_list, (3) the network solve over the grid
  (the step-04 Network, via its `state_provider` seam) -> the SI k(T,P)
  matrix, (4) the EXACT RMG fit of each net reaction's k(T,P) to
  Chebyshev / PDepArrhenius (ported, no substitution), (5) Falloff attach to
  the reactions, (6) write pdep/<network>.yaml (network def + k(T,P) grid +
  fitted coefficients - self-contained, PLAN 12.3). Records wall_solve_s /
  wall_s (the "leverage GPUs" payoff).
- **`rmgpu/kinetics/models.py`** (modified) - the fit path. `Chebyshev`
  gained `fit_to_data` (RMG chebyshev.pyx:177: log10(SI k) on the reduced
  Chebyshev basis, lstsq) and its `__post_init__` c00 unit shift was
  corrected (pre-step-06 hard-coded -6 for ALL models regardless of kunits,
  silently corrupting any SI-fitted Chebyshev; now the RMG kunits-based
  shift via `kunits_to_si`, factor 1.0 for SI so the fit is unaffected).
  `PDepArrhenius` gained the RMG shape (`pressures` + per-pressure
  `arrhenius` list) + `fit_to_data` (an Arrhenius fit per pressure column,
  log-log interpolation between adjacent pressures) + `get_rate_coefficient`
  doing RMG's log-log interpolation; the legacy single-model shape is kept
  for the job-02 assembly tests. `_fit_arrhenius` mirrors arrhenius.pyx:149
  (3-param lstsq on log k).
- **`rmgpu/pdep/network.py`** (modified) - the multi-isomer state seam.
  `Network.__init__` gained `state_provider` (a callable T -> per-T state
  dict); `calculate_rate_coefficients` dispatches state_at ->
  state_provider -> None; `_apply_state` generalized for multi-isomer
  networks (`dens_isomers` (n_isom, n_grains, n_j), per-isomer P_coll and
  coll_freq, Mcoll(T,P) = coll_freq(T,P)*P_coll(T)). No numerics change -
  the CSE machinery was already general over n_isom.
- **`rmgpu/reactor/simulator.py`** (modified) - the loop wiring: `RateParam`
  gained a `falloff` field (the fitted Falloff) + a 'pdep' source;
  `forward_A_TP(rp, T, P)` returns the Falloff's k(T,P) when one is attached,
  else the HPL A_j(T). `simulate_mole_fractions` and `characteristic_rate`
  now use `forward_A_TP` (P was already threaded there - used for c_tot).
- **`rmgpu/core/loop.py`** (modified) - the HPL-stub replacement:
  `RunContext` gained `pressure_dependence` (the YAML block),
  `pdep_reactions` (the reaction_key -> (PDepNetwork, PDepReaction)
  registry built by the production network-state builder, job-07/step-07),
  `pdep_state_provider`; the loop's `_pdep_update` hook (RMG model.py:812-815
  call site - after enlarge, before simulate/screen) runs the driver for the
  registered networks (signature-based invalid marking: re-solve only when a
  network's reaction set changed) and attaches the fitted Falloffs to the
  reactions' RateParams (source='pdep'); wired into `run()`. No-op when no
  block/registry. Module docstring updated (pdep step in the loop).
- **`rmgpu/main.py`** (modified) - the input's `pressure_dependence` block is
  passed into `RunContext` (the real path - the legacy importer does NOT
  carry it; the YAML schema is the seam, see "Open questions" below).
- **`tests/test_pdep_driver.py`** (new, 9 tests) - the wiring test: a
  2-isomer network (RMG-Py-recorded, non-circular) through `run_pdep` -
  (a) Falloff attaches (Chebyshev + PDepArrhenius), (b) pdep/<network>.yaml
  writes + parses (network def + grid + coefficients + k(T,P) grid),
  (c) the fit reproduces the (T,P) grid within the interpolation tolerance
  (log-RMS < 0.5, point-for-point < 100%), plus the non-circular cross-check
  (driver's network solve vs RMG's K_ref) and the loop wiring
  (forward_A_TP + _pdep_update attach/no-op/signature).
- **`gates/baselines/job07/toy_2isomer_ref.json`** (new, ~27MB) - the
  RMG-Py reference for the test (recorded, committed like step 04's
  toy_lindemann_ref.json): the per-T network state (e_list, j_list,
  dens_isomers, dens_product, Kij/Gnj/Fim, eq_ratios, per-isomer P_coll,
  per-P coll_freqs) so rmgpu's Network solves the ME from RMG's state, plus
  RMG-Py's own CSE k(T,P) K_ref on the (T,P) grid + the net-reaction
  (from,to) pairs. 2 isomers + 1 product channel, N2 bath, CSE (Allen),
  grain 3000 J/mol / min 100, 8x6 Gauss-Chebyshev grid (300-800 K, 1e3-1e6
  Pa).
- **`scripts/record_job07_step06_reference.py`** (new) - the recorder
  (rmg_env), reuses step 04's non-circular pattern.
- **`scripts/probe_driver.py`** (new) - end-to-end probe of the draft driver
  on the reference (the K-vs-K_ref + fit-quality numbers below).

## Checks run (commands + real results)

- `/home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest
  tests/test_pdep_driver.py -q` -> **9 passed in 91.52s**.
- `python -m pytest tests/ -q` -> **673 passed in 279.75s** (no regressions
  from the models.py / simulator.py / loop.py / network.py changes).
- `python scripts/record_job07_step06_reference.py` (rmg_env) -> wrote the
  reference (48 solves, ~9 min; the prior session left it 0 bytes - it was
  re-run this session).
- `python scripts/probe_driver.py` (rmgpu env) -> run_pdep OK; n_fitted 4/4;
  **K (network solve) vs K_ref max rel diff 1.1901e-05** (288 nonzero
  entries; CSE precision, not a wiring bug); Chebyshev(6,4) fits log_rms
  0.012-0.027 (max rel vs grid 7.9-34%), PDepArrhenius log_rms 0.036-0.060
  (max rel vs grid 9.3-20.2%) - all under RMG's 0.5 log-RMS error_check bar;
  pdep/toy-2isomer.yaml writes + parses (8x6x3x3 grid, 4 reactions).

Unit-convention settlement (the session-2 open question, via /tmp/probe_fit.py
in rmg_env + reading rmgpy/kinetics/chebyshev.pyx:177 + rmgpy/kinetics/
arrhenius.pyx:149,906): RMG's fit converts the k(T,P) grid to SI before
fitting (Chebyshev: `RateCoefficient(K, kunits).value_si` then log10;
PDepArrhenius: the per-pressure Arrhenius A is stored in the K's units). So
the stored Chebyshev coefficients are log10(SI k), the c00 kunits shift
applies only on file LOAD, and get_rate_coefficient returns SI. rmgpu keeps
SI throughout (the driver passes SI kunits; the shift is a no-op for SI-fitted
coeffs, factor 1e-6/1e-12 for CGS file-load).

## Reference reads beyond the list

- rmgpy/rmg/model.py (the loop call site `update_unimolecular_reaction_networks`
  at :812-815, the network grouping/merge in `add_reaction_to_unimolecular_
  networks` at :1965, `update_unimolecular_reaction_networks` body at :2041)
  - to place the loop wiring at the RMG point (after reactions estimated +
  thermo applied, before simulate) and to model the invalid-network marking
  as a reaction-set signature.
- rmgpy/rmg/input.py - confirmed the method-name normalization (cse ->
  "strong collision" etc.; rs -> "reservoir state") so the driver's alias
  table (and the fix of "randomized strong collision" moving to the RS set)
  matches what the YAML schema produces.
- rmgpy/pdep/network.py - the method dispatch strings (:271-295).

The 3 open questions from the session-1/2 handoff notes, resolved:
1. Where the pressure_dependence block enters rmgpu: the YAML schema
   (`PressureDependenceBlock`, rmgpu/schemas/input.py:456) - the legacy
   importer does NOT carry it (confirmed by grep). It flows
   input -> Input.pressure_dependence -> RunContext.pressure_dependence
   (this step's main.py change) -> the loop's `_pdep_update`.
2. Legacy examples exercising pdep: propane_branching is the gate example
   (per the gate text). The 50 legacy input.py live under RMG-Py (38
   examples/rmg + 12 test/regression); this step's wiring is independent of
   them (the gate, step 07, runs propane_branching).
3. rmgpu/examples/ is absent (the 50 files are in RMG-Py) - noted, no action
   (the examples/ dir in this repo holds the minimal.yaml + handwritten
   minimal).

## Deviations from the step file

- **`run_pdep` signature**: the step file sketched
  `run_pdep(model, pressure_dependence_block, databases)`. The actual
  signature is `run_pdep(network: PDepNetwork, block, method,
  interpolation_model, Tlist, Plist, output_dir, error_check)`. Cause: the
  driver's operating unit is ONE network (RMG's PressureDependenceJob fits
  per-network), and the Network's per-T state (DoS, fluxes, collision) comes
  from the production statmech->DoS->fluxes builder - which is the
  REMAINING numerics piece that the job-07 GATE (step 07, propane_branching)
  builds (RMG-Py network.set_conditions). This step is deliberately the
  ORCHESTRATION/wiring (per the step's own "Pitfalls": "the driver is
  ORCHESTRATION - the numerics are steps 1-5"); the state arrives through the
  step-04/05 `state_at`/`state_provider` seam. The loop's
  `_pdep_update` is the consumer of the driver: it reads
  `ctx.pdep_reactions` (reaction_key -> (PDepNetwork, PDepReaction)) - the
  registry the step-07 state builder will populate - so the wiring is complete
  and the gate plugs its builder into `ctx.pdep_reactions` /
  `ctx.pdep_state_provider` with no further loop change.
- **GPU batching**: the (T,P) grid solve is sequential over the shared GPU
  (the single-GPU-task rule) - the Network's `calculate_rate_coefficients`
  loops T x P in Python. Wall time is recorded (probe: ~12.4s for the
  8x6 2-isomer grid). GPU batch-batching of the ME solves is left to the
  gate (step 07) where the grid is large (propane_branching) and the payoff
  matters; the step file says "where it helps" and for the wiring test it
  does not.
- **`interpolation_model`**: the YAML block stores it as a string
  ("Chebyshev" / "PDepArrhenius"); the driver's `_normalize_interpolation`
  maps it to the fit tuple (Chebyshev defaults to degree 6/4, RMG's default).

## What the next step (job-07/step-07-gate) should know first

1. The driver is READY to be driven on propane_branching: the missing piece
   is the production network STATE builder (statmech -> DoS -> fluxes ->
   collision -> per-T state dict) that RMG-Py's `network.set_conditions`
   does. Build it (port, per PLAN 8a no-QM: E0 from ML thermo, frequencies
   from the statmech DB, TS E0 from the HPL rate via `derive_ts_e0`), wire it
   into `ctx.pdep_reactions` (reaction_key -> (PDepNetwork, PDepReaction),
   each PDepNetwork's `Network.state_provider` set to the builder's
   T -> state), and the loop's `_pdep_update` (already in `run()`) will
   solve + fit + attach. The test
   (tests/test_pdep_driver.py::test_pdep_update_attaches_falloff) shows the
   exact registry shape the gate must build.
2. The gate's k(T,P) parity (the <1% target) is the numerics check - this
   step's probe shows the driver's wiring reproduces RMG's K_ref to 1.2e-5
   (CSE precision) on the 2-isomer reference, so the driver is not the source
   of any gap; the gap (if any) will be in the step-07 state builder
   (grains / DoS / collision) or the gate's own RMG-Py reference.
3. The grid + fit are exact ports (Gauss-Chebyshev grid matches the recorded
   grid to 1e-12; the fit is RMG's). The gate should use the block's
   interpolation_model (Chebyshev 6/4 default) to match RMG's fit.
4. The CSE method only: MSC/RS/SLS raise NotImplementedError (job 08). The
   gate is CSE-only (per the gate text).
5. Unit convention: SI throughout (the fit stores SI; the c00 shift is a
   file-load-only artifact). The gate's RMG-Py reference must extract the SI
   k(T,P) matrix the same way (RMG's core-loop path does this internally).
6. The reference JSON is committed (27MB) - if the gate regenerates
   propane_branching's reference, follow the same pattern (record, commit,
   validate rmgpu against it).
