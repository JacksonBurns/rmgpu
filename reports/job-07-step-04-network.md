# job-07/step-04: The pdep network + master equation (CSE) + TS-E0

Date: 2026-09-06
Status: GREEN (all checks pass; committed)
Commits: d87120c

## What was built

- `rmgpu/pdep/network.py` (new, ~640 lines): the pressure-dependent
  unimolecular reaction network, ported from RMG-Py
  (`rmgpy/pdep/{network.py,me.pyx,cse.pyx,reaction.pyx}` + `rmgpy/rmg/pdep.py`).
  Contents, one line each:
  - `Network` class: the state container (isomers/reactants/products, e_list,
    j_list, dens_states, eq_ratios, Kij/Gnj/Fim, coll_freq, Mcoll, E0, E0_ts,
    grain params) + the step-04-owned computations.
  - `select_energy_grains` / `_get_energy_grains`: grain generation (max grain
    size / min grain count / energy_grid logic). e_max = max(max channel E0,
    TS E0s) + 40kT (RMG includes the TS energy - this was a port bug found +
    fixed).
  - `calculate_equilibrium_ratios`: the isomer equilibrium ratios.
  - `derive_ts_e0` + `network_energy_correction`: the no-QM TS E0 derivation
    (the PLAN 8a.2 path).
  - `generate_full_me_matrix`: the full master-equation matrix (collision +
    isomerization + association/dissociation terms; the me.pyx port).
  - `apply_cse_allen`: the CSE (Allen) k(T,P) extraction (the cse.pyx port,
    incl. RMG's zero-k(T,P)-when-eigenvalues-do-not-separate behavior + the
    symmetrization check).
  - `apply_ilt_k_e`: the ILT microcanonical rate (the no-QM rate, reaction.pyx
    port; the convolve helper is ported too).
  - `calculate_rate_coefficients`: the (T,P)-grid driver + method dispatch on
    the long strings. CSE (allen) = this step; MSC/RS/SLS/georgievskii raise
    NotImplementedError (job-08).
  - `seed_collision(coll_freq, Mcoll)`: the collision SEAM - step-05's
    collision model plugs in here without touching this module.
- `rmgpu/pdep/__init__.py`: package exports.
- `tests/test_pdep_network.py`: 6 tests (TS-E0, two-reactant TS-E0 sanity,
  grain parity, CSE k(T,P) parity, ILT k(E) parity, method dispatch).
- `scripts/record_job07_step04_reference.py`: the RMG-Py (rmg_env) recorder -
  non-circular. Builds a toy Lindemann network in RMG-Py (1 isomer + 1
  dissociation channel + 1 path reaction via the ILT path, CSE allen) and dumps
  the per-T state + RMG-Py's own CSE k(T,P) (K_ref) + RMG-Py's raw ILT k(E) to
  the reference.
- `gates/baselines/job07/toy_lindemann_ref.json` (2.3 MB): the recorded
  reference (RMG-Py's own run).

## Checks run (the commands + the real results)

All run with the rmgpu env interpreter
(`/home/jackson/miniforge3/envs/rmgpu/bin/python`). The reference is generated
with the rmg_env interpreter
(`/home/jackson/miniforge3/envs/rmg_env/bin/python
scripts/record_job07_step04_reference.py`).

1. `pytest tests/test_pdep_network.py -q` -> **6 passed** (1.0s).
2. TS-E0 derivation (the no-QM path): `derive_ts_e0` reproduces RMG-Py's derived
   E0_TS from the reference (abs 1e-9); `network_energy_correction` exact
   (abs 0.0). This is the PLAN 8a.2 proof.
3. Grain grid vs RMG-Py: counts + boundaries EXACT (max diff 0.0) at T=400, 600
   vs RMG-Py's initial (pure) `select_energy_grains` (e_list_initial in the
   reference).
4. ME integration / k(T,P): the toy Lindemann k(T,P) via CSE (Allen) vs
   RMG-Py's recorded K_ref - **max rel diff 1.9e-11** (machine precision;
   8/8 (T,P) points nonzero). This is the sub-gate that de-risks the job-07
   k(T,P) gate.
5. ILT k(E) port: rmgpu's ILT k(E) vs RMG-Py's raw ILT k(E) - **max diff 0.0**
   (T=400, 600); zero where RMG has zero.
6. Method dispatch: CSE (allen) works; MSC/RS/SLS/georgievskii raise
   NotImplementedError; unknown method raises PDepNetworkError.
7. Full suite: `pytest tests/ -q` -> **656 passed** (was 650; +6 pdep tests,
   no regressions; 3:08).

## Reference reads beyond the list

The step file lists network.py, me.pyx, reaction.pyx, arkane/pdep.py. Beyond
that, `rmgpy/pdep/cse.pyx` (the CSE/Allen k(T,P) extraction - the actual
deliverable; the brief's "network.py's extraction" is the dispatch, the math
is in cse.pyx), `rmgpy/rmg/pdep.py` (lines ~745-870: the no-QM TS-E0
derivation + energy_correction + the core-loop network setup - the ACTUAL
no-QM path), and `rmgpy/pdep/collision.pyx` (the SingleExponentialDown
collision - to define the step-05 SEAM, not to port it). The job-06 reactor
torchdae backend was read (for the ME-integration choice), but the step-04 CSE
path uses the eigen-decomposition, not the time-dependent ME ODE (the ODE is
the SLS path, job-08) - so no torchdae integration was needed this step.
`test/rmgpy/pdep/networkTest.py` was read for the toy-network recipe (hand-set
conformers + SingleExponentialDown + CSE).

## Deviations from the step file

1. **TS-E0 form**: the step brief cites the TST form
   `E0_TS = sum(reactant E0) - R*T*ln(k_inf*V/h)` (citing arkane/pdep.py).
   RMG-Py's ACTUAL no-QM core-loop path (`rmgpy/rmg/pdep.py:856`) implements
   the **Ea-based form** `E0_TS = sum(reactant E0) + Ea + energy_correction`.
   The core loop has no `k_inf`/`V` available at that point - it derives the
   TS E0 from the reaction's Arrhenius Ea. Since the job-07 gate sub-gate 2
   compares rmgpu's E0_TS to RMG-Py's within 1e-8 (a BLOCKER per PLAN 8a.2),
   `derive_ts_e0` ports RMG-Py's exact Ea-based expression. The k_inf*V/h TST
   form is documented as the equivalent at the HPL limit (up to T^n and
   tunneling corrections RMG folds into the Arrhenius fit) but is NOT what
   RMG-Py's core loop uses.
2. **Toy Lindemann, not propane_branching**: the step-04 scope is the Network +
   ME + CSE + TS-E0 in isolation; the toy case de-risks the job-07 k(T,P) gate
   per the step brief ("this test must pass before the driver is wired"). The
   full propane_branching k(T,P) parity (sub-gate 3) is the job's GATE step
   (07/07), which runs the real example.
3. **Seeded DoS + collision matrix**: the step-04 unit test seeds the DoS
   (step-01/03's, currently a stub) and the collision matrix (step-05's,
   unbuilt) from the RMG-Py reference, isolating the step-04-owned
   grain/ME-matrix/CSE/TS-E0 logic. This is non-circular: every compared value
   is RMG-Py's own run, never rmgpu's. The step-05 collision model + step-03
   conformer assembly will replace the seeding in step-06 (driver wiring).
4. **No torchdae ME integration this step**: the CSE (Allen) method solves the
   ME by eigen-decomposition (cse.pyx), not by integrating the stiff time-
   dependent ME ODE (that is the SLS path, job-08). The step brief lists the
   torchdae ME integration as a deliverable, but it is the SLS path's
   integration, not the CSE Allen path's - so it is deferred to job-08 with the
   SLS method. The CSE Allen path (this step) is complete and parity-verified.

## What the next step (step-05 collision) should know first

- The `Network.seed_collision(coll_freq, Mcoll)` is the SEAM. Step-05 builds
  `rmgpu/pdep/collision.py` (the SingleExponentialDown port) to COMPUTE
  `coll_freq` + `Mcoll` from the bath gas + energy-transfer model, and the
  pdep driver (step-06) calls it instead of seeding. The step-04 CSE
  extraction (`apply_cse_allen`) consumes `Mcoll` verbatim, so it plugs in
  without touching `network.py`.
- RMG's collision matrix factors as `Mcoll(T,P) = coll_freq(T,P) * P_coll(T)`.
  `P_coll` (the collision probability matrix) is P-independent; `coll_freq` is
  ~P. The reference stores them separately (P_coll per-T, coll_freqs per-(T,P))
  to keep it compact - step-05 should reproduce that factorization so the
  reference can be reused to validate step-05's collision model.
- RMG-Py's runtime grain selection is the `set_conditions` k(E)-validity retry
  loop (halves the grain size until the ILT k(E) reproduces the HPL limit), NOT
  the pure `select_energy_grains`. The reference records both: `e_list_initial`
  (pure, = the step-04 grain-generation port's target) and `e_list` (post-retry,
  = what the CSE test seeds). Step-05/06 should use the post-retry grid (the
  `e_list`), matching RMG-Py's runtime behavior.
- The CSE eigen-value separation (the `n_cse != n_chem` branch) returns a ZERO
  k(T,P) matrix in RMG-Py (no lumping order passed). The rmgpu port matches
  that exactly (prints + returns zero) - do not "fix" it into a fallback, the
  gate must see the same zero as RMG-Py.
