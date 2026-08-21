# job-07: Statmech + master equation (CSE) + pdep gate

Read ORIENTATION.md (master equation data path - re-read it) and PLAN.md sections 4
(retain #5/#6), 8a (the full explanation), 10 (Phase 3). Prereqs: jobs 01-06.
This is the numerically hardest job in the project. Take the time; port CSE first
and gate it before touching MSC/RS/SLS.

## Goal

Pressure dependence for unimolecular/reaction networks: statistical mechanics
(density of states from vibrational/rotor/translation modes), the discretized
master equation (CSE lumping), collision models, and the (T,P)-grid solving +
Chebyshev/PDepArrhenius fitting that produces `Falloff` kinetics objects for the
rate registry. NO QM anywhere (PLAN.md 8a): E0 from ML thermo, frequencies from the
statmech DB, TS E0 derived from the HPL rate.

## Reference (read the actual code; the .pyx files are the spec)

  RMG-Py/rmgpy/statmech/           - conformer.pyx, mode.pyx, vibration.pyx,
    rotation.pyx (linear/nonlinear + hindered), torsion.pyx (1D rotor PDE),
    ndTorsions.py (2D), translation.pyx, schrodinger.pyx, mode.pxd.
    Port to numpy/torch. The 1D/2D rotor eigenproblems go to scipy.linalg /
    torch.linalg (document the choice).
  RMG-Py/rmgpy/data/statmech.py (32k) - the STATMECH DATABASE path (group
    frequencies get_statmech_data, lines ~318-460; GroupFrequencies; the
    fit_statmech_to_heat_capacity path from data/statmechfit.py). This is how
    rmgpu gets frequencies WITHOUT QM. Port it (it is database + fitting logic,
    not estimation).
  RMG-Py/rmgpy/pdep/
    network.py (61k) - the network class: grains, the rate matrix (RRKM k(E) from
      TS DoS + energy gap), the time-dependent master equation integration, the
      k(T,P) phenomenological rate extraction, grain generation.
    cse.pyx (17k) - CSE collision model (the default). Port FIRST.
    collision.pyx, configuration.pyx - grain configuration + collision frequency
      (calculate_collision_frequency).
    me.pyx - the time-dependent ME ODE (port to torchdae: it is a stiff ODE in the
      grain populations - use the torchdae backend from job 06).
    reaction.pyx - the pressure-dependent reaction object (isomers/channels).
    msc.pyx, rs.pyx, sls.py - the other lumping methods (job 08 - port them there).
  RMG-Py/rmgpy/rmg/pdep.py (46k) - the PDEPENDENCE DRIVER (how RMG decides which
    reactions get pdep, builds networks, the (T,P) grid from the input's
    pressure_dependence block, fitting to Chebyshev/PDepArrhenius). Port this
    orchestration (it connects the core loop's pdep stub from job 06 to the engine).
  RMG-Py/arkane/pdep.py (lines ~270-350) - the TS E0 DERIVATION from the HPL rate
    (the no-QM path): E0_TS = sum(reactant E0) - R*T*ln(k_inf*V/h). Port exactly,
    including the tunneling parameter handling (keep Wigner from job 04).
  RMG-Py/rmgpy/kinetics/falloff.pyx + chebyshev.pyx - already ported in job 04
    (registry); this job PRODUCES their parameters.

## Deliverables

1. `rmgpu/statmech/` - ported statmech:
   - Conformer (E0, spin multiplicity, modes list), HarmonicOscillator,
     LinearRotor/NonlinearRotor, HinderedRotor, FreeRotor (1D, 2D via
     ndTorsions), Translation.
   - `get_heat_capacity(T)`, density of states `rho(E)` (the RRKM DoS from
     frequencies + rotors - port the standard convolution / recursion RMG uses;
     numpy; this is hot - vectorize, but correctness first),
   - conformer assembly from the statmech DB (job: port data/statmech.py's
     get_statmech_data logic: group frequencies + heat-capacity fitting for the
     remainder, per RMG-Py data/statmechfit.py).
   - symmetry factors (job-01 symmetry.py) wired into the DoS.
2. `rmgpu/pdep/network.py` - the network + master equation:
   - Network: isomers, reactant/product channels, bath gas, grains
     (generate from RMG's grain rules: max grain size, min count, the
     energy_grid logic from network.py), the rate matrix (RRKM k(E) diagonal,
     collision off-diagonal via CSE), the time-dependent ME integration via
     torchdae (me.pyx port), k(T,P) extraction (the phenomenological rate
     from the quasi-stationary flux - port network.py's extraction logic).
   - TS E0 derivation from HPL rate (arkane/pdep.py lines 273-282) - THE no-QM
     path; test it standalone first (a known Lindemann case must reproduce
     RMG's k(T,P)).
3. `rmgpu/pdep/collision.py` - CSE (first), collision frequency
   (calculate_collision_frequency from transport LJ params - job-02 TransportDB),
   + collision.pyx/configuration.pyx grain logic.
4. `rmgpu/pdep/driver.py` - port rmg/pdep.py orchestration: select pressure-
   dependent reactions (from family metadata), build networks (grouping isomers
   per RMG's network rules), run the (T,P) grid (from the YAML
   pressure_dependence block: Tmin/Tmax/Tcount, Pmin/Pmax/Pcount), fit k(T,P)
   to Chebyshev/PDepArrhenius, attach Falloff to the reactions' rate models
   (the registry from job 04), and write pdep/<network>.yaml (PLAN.md 12.3) with
   the network definition + k(T,P) grid.
5. Wire into the core loop: replace job-06's HPL stub so pressure-dependent
families get Falloff kinetics in the mechanism (the loop's simulate step then
uses k(T,P) at the reactor conditions).

NOTE on method strings (verified in RMG-Py/rmgpy/pdep/network.py
`calculate_rate_coefficients`): the method field is a LONG descriptive string,
not a shorthand. Accepted values (port the dispatch exactly):
  'modified strong collision'  -> MSC
  'reservoir state'            -> RS
  'chemically-significant eigenvalues'  -> CSE (allen)
  'chemically-significant eigenvalues georgievskii' -> CSE (georgievskii)
  'simulation least squares' / 'simulation least squares ode' /
  'simulation least squares matrix exponential'  -> SLS (mexp/ode/mexp)
The YAML `pressure_dependence.method` (job-03 schema) should accept both the
shorthand (cse|masc|rs|sls) AND the long strings, normalizing shorthand to the
long form. Default to 'chemically-significant eigenvalues' (CSE) per RMG.

## Gate (job 07) -> gates/gate_07.py, report reports/job-07.md

  1. Statmech unit parity: for 50 species/intermediates from the propane_branching
     mechanism: conformer assembly (E0, spin, mode counts) matches RMG-Py's
     (generate reference via a small RMG-Py script using the same statmech DB);
     heat capacity Cp(T) on a 300-1500K grid matches within 1e-6 relative;
     density of states rho(E) on a grid matches RMG-Py's DoS (script) within
     1e-6 relative (spot-check 10, report max diff).
  2. TS-E0 derivation: for 20 pressure-dependent reactions: E0_TS (rmgpu, from
     HPL rate) == E0_TS (RMG-Py) within 1e-8 (this pins the no-QM path).
  3. CSE k(T,P) parity (THE gate): run the propane_branching example
     (examples/rmg/propane_branching) with pressure_dependence method=cse in BOTH
     rmgpu and RMG-Py (same T/P grid from its input). For every pressure-
     dependent reaction: compare the fitted Falloff k(T,P) on the grid - report
     max relative diff in k_inf, k_0, and the broadening params; also the
     Chebyshev/PDepArrhenius coefficients. TARGET: < 1% relative on k(T,P)
     values, < 1e-3 on coefficients (document tolerance achieved). Any >1%
     reaction: list it + diagnose (grain mismatch? collision model? DoS?).
  4. Network parity: network structure (isomers, channels, grain counts) matches
     RMG-Py for propane_branching's networks (counts + a spot-check grain grid).
  5. The full rmgpu run of propane_branching (with pdep ON) completes; compare
     final core species/reaction counts vs RMG-Py's (expect small divergence vs
     job-06's HPL run - that is expected and fine; the POINT is the k(T,P) parity
     above).
  A RED gate: port a second family's example (c3h4 has pdep reactions) to confirm
  it is not propane-specific.

## When done

STATUS.md (job 07 row + log entry with the k(T,P) max-diff numbers) + commit
"job-07: statmech + master equation (CSE) + pdep parity" + STOP.

## Pitfalls

- The DoS + RRKM k(E) is the numerical core; a subtle bug (symmetry factor,
  ZPE, grain boundaries) silently shifts k(T,P). The gate's DoS/TS-E0 sub-gates
  exist to catch these BEFORE the k(T,P) comparison. If k(T,P) disagrees but DoS
  matches, look at grains/collisions; if DoS disagrees, look at conformer assembly.
- torchdae for the ME time integration: the ME is stiff (collision frequencies
  vary by orders of magnitude across grains). Use TR-BDF2 or BDF2; validate
  against RMG-Py's integration (which uses a Python ODE solver) - the k(T,P)
  gate covers this.
- The (T,P) grid can be large (Tcount x Pcount x reactions x networks). This is
  where GPU matters (batch the (T,P) grid solves) - but do NOT parallelize across
  the shared GPU with llama-server; run sequentially, batch within a solve.
  Record wall-time in the report (this is the "leverage GPUs" payoff - show it).
- Chebyshev vs PDepArrhenius fitting: port RMG's exact fit (the input's
  interpolation_model picks; default per RMG). Do not substitute a different fit.
- Do NOT port MSC/RS/SLS this job (job 08). CSE only. The driver must select
  method from the YAML and raise NotImplemented for MSC/RS/SLS until job 08.
