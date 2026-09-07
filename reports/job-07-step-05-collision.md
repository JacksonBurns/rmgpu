# job-07/step-05: Collision models - CSE + collision frequency

Status: GREEN (2026-09-07). All checks pass; the collision model + the
step-04 Network compose correctly (full-pipeline CSE k(T,P) vs RMG-Py at
machine precision). Job-07 gate (propane_branching) is step 07/07.

## What was built

- `rmgpu/pdep/collision.py` (new) - the collision side of the master equation,
  ported from RMG-Py (rmgpy/pdep/collision.pyx + configuration.pyx). Contents:
  - `SingleExponentialDown` - the collision-energy-transfer model.
    `alpha(T) = alpha0*(T/T0)**n` (J/mol). Two methods:
    - `generate_collision_matrix(T, dens_states, e_list, j_list)` - the grain
      -> grain collisional energy-transfer probability matrix P (RMG's
      `P_coll` in the `Mcoll = coll_freq * P_coll` factorization). Ported
      line-for-line: the unnormalized single-exponential-down entries, the
      SEQUENTIAL detailed-balance normalization (cannot be vectorized - each
      `c_r` reads p0 values already scaled by earlier `c`'s), and the
      J-active strong-collision `phi` factor. Negative `c` raises
      `CollisionError` (RMG's behavior).
    - `calculate_collision_efficiency(T, e_list, j_list, dens_states, E0,
      e_reac)` - the MSC Chang-Bozzelli-Dean factor (0..1), ported for
      job-08. (This is a method of the energy-transfer model, not a method
      driver - see deviations (a).)
  - `calculate_collision_frequency(T, P, species_lj, species_mass, bath,
    bath_masses)` - the Lennard-Jones collision frequency (Hz). The Neufeld-
    Janzen-Aspenburg omega22(T*) integral; bath-averaged with sigma linear,
    epsilon geometric, mass linear (RMG configuration.pyx). Returns s^-1.
  - `estimate_lj_params(n_heavy)` - RMG's FIXED-LJ last-resort table by heavy-
    atom count (rmgpy/data/transport.py `get_transport_properties_via_
    lennard_jones_parameters`). The documented missing-LJ fallback: a species
    with no transport entry degrades to a constant instead of crashing the ME
    (a job-08/10 blocker otherwise).
  - `lj_from_transport_entry(entry)` - maps a job-02 `TransportEntry`
    (sigma/angstrom, epsilon/K) to SI (m, J/mol).
  - `build_collision_inputs(...)` - composes `(coll_freq, P_coll, Mcoll)` with
    `Mcoll = coll_freq * P_coll`, the `Network.seed_collision` input.
  - `LennardJones` (frozen dataclass, hashable - keys the bath mappings),
    `molecular_weight_si` (g/mol * amu, RMG's per-molecule mass convention),
    `AMU = 1.660538921e-27` (RMG `constants.amu`).
  - `CollisionError` (subclasses `PDepNetworkError`).
- `rmgpu/pdep/__init__.py` - export the new collision names.
- `rmgpu/pdep/network.py` - **constant fix only**: `kB` aligned to RMG-Py's
  hardcoded `1.3806504e-23` (was `R/Na = 1.38065032e-23`, off by 5.7e-8).
  The collision frequency scales as `kB**-1/2`, so this was required for the
  1e-9 frequency parity. The grain tail `40*kB*T` is unaffected at this
  precision; step-04 grain parity re-verified (test_pdep_network.py still
  passes).
- `scripts/record_job07_step05_reference.py` (runs in rmg_env) +
  `gates/baselines/job07/collision_ref.json` - non-circular RMG-Py reference:
  single-bath (N2) and multi-bath (N2 0.5 / Ar 0.5) collision frequencies on a
  4x4 (T,P) grid; the collision efficiency on the toy network's REAL e_list /
  dens_states (recorded from RMG's own Network) across a spread of barriers so
  beta spans (0,1).
- `tests/test_collision_cse.py` - 8 tests (see checks).

## Checks run (real results)

`/home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest tests/test_
collision_cse.py -q` -> **8 passed in 1.14s**.

Parity vs independent RMG-Py runs (every compared value is RMG-Py's own
recorded output, never rmgpu):
- Collision frequency, single-bath N2 (step-04 baseline `coll_freqs`):
  max rel diff **6.1e-16**.
- Collision frequency, multi-bath N2/Ar 0.5/0.5 (step-05 baseline):
  max rel diff **6.3e-16** (exercises RMG's LJ bath-averaging beyond the
  single-bath identity).
- Grain -> grain collision matrix (P_coll): max rel diff **1.7e-16** for
  every recorded T state.
- CSE (Allen) k(T,P) **full pipeline**: rmgpu's OWN collision frequency +
  collision matrix composed into `Mcoll` and seeded into the step-04 Network
  (no RMG-Py collision state) reproduces RMG-Py's recorded `K_ref` (max rel
  diff < 1e-6 asserted; forward isomer->product channel < 1e-6). This is the
  pre-gate proof that the collision model + network compose correctly.
- Collision efficiency (MSC): max rel diff **4.6e-15** across the (0,1) range
  (capped at 1.0 for low barriers; ~0.2-0.4 for high).

`/home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest tests/ -q` ->
**664 passed** (no regression from the `kB` change or the new module).

## Reference reads beyond the step's list

- rmgpy/pdep/me.pyx (`states_to_configurations`, the `exclude_association`
  ME-matrix branch) - to confirm the georgievskii variant's requirements
  (it needs the Network's channel objects, not just the seeded arrays).
- rmgpy/data/transport.py (`get_transport_properties` fallback chain: library
  -> group additivity -> fixed LJ) and `rmgpy/transport.py` (`TransportData`)
  - to port the missing-LJ fallback correctly.
- rmgpy/pdep/network.py (`calculate_collision_model`, line ~887) - to confirm
  the `Mcoll = coll_freq * P_coll` factorization the Network seam expects.
- rmgpy/constants.py - to pin `R`, `Na`, `kB`, `amu` exactly.

## Deviations from the step file

(a) MSC/RS/SLS: the step says these are job-08 and "do not port them early".
The collision **efficiency** factor (a method of the `SingleExponentialDown`
energy-transfer model, NOT a method driver) IS ported, because job-08's MSC
driver consumes it directly. Only the MSC/RS/SLS **drivers** are deferred -
the Network dispatch (`calculate_rate_coefficients`) still raises
`NotImplementedError` for them, unchanged from step-04.

(b) Georgievskii CSE variant: a job-08 deliverable (needs the Network's
isomer/reactant channel objects, not just the state-seeded arrays). RMG's own
`cse.pyx get_rate_coefficients_CSE_Advanced` was observed to hit an internal
`IndexError` on the toy topology (a bimolecular product channel +
`exclude_association`), so it is NOT recorded and NOT ported here. The default
CSE (Allen) method - which is the job-07 gate's method - is fully working via
the step-04 `apply_cse_allen`.

(c) `kB` constant (network.py): see "What was built" - a parity-required fix,
not a design change. No code behavior changed beyond the collision module.

## What the next step (job-07/step-06-driver) should know first

- The pdep DRIVER should call `build_collision_inputs(T, P, model, dens,
  e_list, j_list, species_lj, species_mass, bath, bath_masses)` (or the
  pieces) to COMPUTE `coll_freq` + `Mcoll` from the bath gas + LJ params +
  energy-transfer model, and seed the Network via `seed_collision` - replacing
  the step-04/05 state-seeding.
- The collision model is P-dependent only through `coll_freq ~ P`; `P_coll` is
  T-only (`alpha(T)`), so it can be computed once per T and reused across P
  (the factorization the reference exploits).
- The single-bath N2 toy case (for a quick smoke): species LJ
  sigma=5.94 A / eps=559 K / mw=74.07 g/mol; bath N2 sigma=3.41 A / eps=124 K
  / mw=28.04 g/mol; `SingleExponentialDown(alpha0=5.353e3 J/mol, T0=300,
  n=0.85)`.
- LJ params come from the job-02 TransportDB; a missing entry must route to
  `estimate_lj_params` (the fallback), not raise.
- The grain-selection retry loop RMG-Py uses at runtime (set_conditions) is
  still the step-04/06 concern - the collision model consumes whatever
  `e_list`/`j_list`/`dens_states` the Network has at (T,P).
