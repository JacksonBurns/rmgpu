# job-12: Catalysis plugin (post-parity; the heavy plugin)

Read ORIENTATION.md and PLAN.md section 9 (9.1 protocol, 9.2 catalysis) in full.
Prereqs: job-11 (plugin protocol validated by solvation). This is the big plugin:
a second phase (surface) with its own species type, families, thermo/kinetics
providers, and a coverage-tracking reactor. PLAN.md is explicit: we do NOT intend
to build this for the core-parity PoC - but this job is the design-validated
implementation, done only after gas+liquid parity.

## Goal

A `rmgpu/plugins/catalysis/` package that makes surface-catalyzed mechanism
generation work through the EXISTING core (recipe engine, ML estimators, reactor
interface) without forking any core module. The core sees only the plugin
protocol.

## Reference (read - the RMG-Py surface code is the spec for what a full
implementation looks like; PLAN.md 9.2 maps each piece)

  RMG-database/input/kinetics/families/Surface_*/   - the ~60 surface families
    (recipes: adsorption single/double/vdW/dissociative, dissociation, migration,
    abstraction, Eley-Rideal / Langmuir-Hinshelwood, beta-scission,
    proton/electron transfer, ...). These are recipe DATA - they flow through the
    job-05 recipe engine.
  RMG-database/input/surface/ (or input/surface/libraries - locate it) - the
    adsorbate library (adsorption energies), surface site definitions.
  RMG-Py/rmgpy/kinetics/surface.pyx (56k) - surface rate expressions
    (coverage-dependent pre-exponential, site-balance terms, BEP relations).
    Port onto the job-04 rate registry (the surface models carry extra
    coverage-theta terms).
  RMG-Py/rmgpy/solver/surface.pyx (55k) - the SurfaceReactor: tracks site
    coverage (vacant vs occupied) as extra state + the site-balance ALGEBRAIC
    constraint (this is the DAE index-1 case torchdae handles). Port onto the
    torchdae backend.
  RMG-Py/rmgpy/data/thermo.py - `correct_binding_energy` /
    `set_binding_energies` (linear-scaling correction of adsorption energies
    against a reference metal) - the coverage-dependent thermo provider.
  RMG-Py/rmgpy/species.py - SurfaceSpecies = adsorbate + site (the site is a
    small site-graph: a surface atom + neighbors). Port the representation.
  Examples: examples/rmg/minimal_surface, minimal_multisurf, liquid_cat (cat part),
    test/regression/minimal_surface, RMS_liquidSurface_ch4o2cat.

## Deliverables

1. `rmgpu/plugins/catalysis/` package:
   - `species.py`: `SurfaceSpecies` (adsorbate + site-graph) - registered via
     `register_species` so the core loop can hold it like any species. The site
     graph is a small Molecule (metal atom + neighbors) with facet info.
   - `families.py`: load the Surface_* family recipes from the DB (job-02/05
     machinery - they are recipes; the plugin only supplies the site templates +
     the surface-reactant patterns). NO new generator - the job-05 recipe engine
     runs them.
   - `thermo.py`: adsorption-energy thermo provider: for a surface species,
     adsorption energy (from the adsorbate library / ML adsorption model) +
     optional coverage-dependent + linear-scaling correction (port
     correct_binding_energy).
   - `kinetics.py`: surface rate-model providers (coverage-dependent, BEP) wired
     into the job-04 registry (the registry gets a `SurfaceArrhenius`/
     `SurfaceArrheniusBEP` model type - port the math from surface.pyx; the
     existing registry stays, these are new model types).
   - `reactor.py`: SurfaceReactor (coverage state + site-balance algebraic
     constraint via torchdae) registered via `register_reactor`.
   - `plugin.py`: the `RmgpuPlugin` implementation tying it together
     (register_*, the hooks, load_database pulling the surface/adsorbate tables
     from rmgdb - if rmgdb does not yet have a surface DB, document it and read
     the raw RMG-database surface files for just that, per job-02's gap rule).
2. YAML: the `reactors:` surface type (job-03 schema stub) becomes live; the
   `database:` block gains surface library/family selection when a surface reactor
   is present.
3. `reports/job-12.md`: the implementation mapped to PLAN.md 9.2 (each bullet of
   9.2 -> where it lives in the plugin), + any protocol change forced by reality
   (if any, update PLAN.md 9.1).

## Gate (job-12) -> gates/gate_12.py, report reports/job-12.md

  1. SurfaceSpecies round-trip: site-graph construction + adsorbate representation
     parity with RMG-Py (a fixed set of sites/adsorbates from the examples:
     structure, facet, label).
  2. Surface family application: for 20 (site, adsorbate/gas-species) cases:
     product enumeration (via the job-05 engine + the plugin's families) matches
     RMG-Py's (products + degeneracy).
  3. Coverage-dependent kinetics: for 15 surface reactions: k(coverage, T) from
     the plugin's rate models == RMG-Py's (surface.pyx) within 1e-8.
  4. Surface reactor: run `minimal_surface` in rmgpu vs RMG-Py: mechanism sets
     (core species/reaction Δ) + coverage profiles (vacant-site fraction vs time)
     max diff. The site-balance constraint must be satisfied to solver tolerance
     at every step (assert it in the gate - this is the DAE correctness check).
  5. Plugin isolation: catalysis code lives ONLY in rmgpu/plugins/catalysis/ +
     the registered hook call-sites; a gas-phase run with the plugin INSTALLED but
     no surface reactor is unchanged (the plugin is inert without a surface
     reactor).

## When done

STATUS.md + commit "job-12: catalysis plugin" + STOP.

## Pitfalls

- The site-balance constraint is the DAE's algebraic part; torchdae's index
  reduction must handle it (PLAN.md 9.2 calls this out). If torchdae struggles
  with the constraint, that is a torchdae finding to record (PLAN.md 13 risk 3),
  not a reason to fork the solver.
- Coverage dependence couples thermo and kinetics (adsorption energy shifts with
  coverage; rates carry theta terms). Keep the providers separated (thermo
  provider returns the corrected energy; kinetics provider carries the theta
  terms) - the core composes them.
- Multi-surface (minimal_multisurf) - several sites at once - is the hard case;
  gate the single-surface case first, then multi-site.
- Do NOT add surface species to the gas-phase core loop when no surface reactor
  is present (plugin inertness, gate 5).
