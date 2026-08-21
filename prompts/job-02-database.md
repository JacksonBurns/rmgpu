# job-02: Database layer via rmgdb

Read ORIENTATION.md (External: rmgdb) and PLAN.md section 5. Prereq: job-01.

## Goal

ALL database I/O for rmgpu goes through rmgdb - the SQL/SQLite wrapper over
RMG-database that the user maintains. No YAML-parsing of RMG-database files inside
rmgpu (except for data that rmgdb does not cover yet, which must be documented, not
silently worked around).

## Context (verified)

- rmgdb source: /home/jackson/rmgpu/rmgdb
  - install: cd /home/jackson/rmgpu/rmgdb/standard && pip install -e .  (rmgdb pkg)
    then cd ../data && pip install -e . (rmgdatabase, the builder)
  - built SQLite: /home/jackson/rmgpu/rmgdb/db/{thermo,kinetics,transport,solvation,statmech}.db
  - demo of the query API: /home/jackson/rmgpu/rmgdb/demo.ipynb - READ IT, it is the
    canonical usage.
  - schemas: /home/jackson/rmgpu/rmgdb/standard/rmgdb/{thermo,kinetics,transport,solvation,statmech}/schema.py
- Raw RMG-database (for anything rmgdb misses): /home/jackson/rmgpu/RMG-database/input/
  - kinetics families: input/kinetics/families/<Name>/family.py (+ templates, rate rules YAML)
  - kinetics libraries: input/kinetics/libraries/
  - thermo libraries: input/thermo/libraries/  (77 of them)
  - statmech: input/statmech/{groups,libraries,depository}
  - transport: input/transport/{groups,libraries}
  - solvation: input/solvation/...
  - forbidden structures: input/forbiddenStructures.py
  - recommended library sets: input/recommended_libraries.yml

## Deliverables

1. `rmgpu/db/loaders.py` (or a small db/ package):
   - `ThermoDB` facade: entry lookup by structure (substructure match against library
     entries, using job-01 Molecule + rmgdb thermo tables), label lookup,
     "is this in library X" checks, entry count. Returns rmgpy-compatible-semantic
     data objects (see 3).
   - `KineticsDB` facade: same for kinetics libraries + depository + rate rules +
     family definitions (families: recipes, templates, rate rules - the rmgdb
     kinetics schema stores all of it; verify what's in kinetics.db by querying it,
     and document any gap).
   - `TransportDB`, `StatMechDB`, `SolvationDB` facades (minimal: what later jobs
     need - transport in job 06/07, statmech in job 07, solvation in job 11).
   - a `Databases` aggregate object: constructed from a config (library lists, family
     list - mirrors the YAML `database:` block from job 03), caching loaded data.
2. `rmgpu/data/entries.py` (or similar): data-only classes matching RMG semantics:
   ThermoEntry (Wilhoit or NASA7 coefficients + T bounds + reference),
   KineticsEntry (one of the rate models below + reference), with `.to_yaml()`-style
   serialization for the mechanism artifact (job 03/06).
   Rate models (port the MATH from the .pyx, numpy/torch only, NO Cython):
     Arrhenius, ArrheniusEP (with Tmin/Tmax), ArrheniusBM (Bayesian-modeling form,
     only as a storage type - the BM estimator itself is deleted), Marcus,
     Lindemann/Troe/ThirdBody/Falloff (the k(T,P) wrapper + Troe broadening params),
     Chebyshev + PDepArrhenius (the pdep interpolation models).
   Reference files to read for the math:
     RMG-Py/rmgpy/kinetics/arrhenius.pyx (87k), falloff.pyx, chebyshev.pyx,
     model.pyx (the model registry: forward/reverse rate generation,
     generate_reverse_rate_coefficient - thermodynamic consistency via
     dG(T) integration), kineticsdata.pyx, tunneling.pyx (Wigner only; the rest
     is used by job 09? no - tunneling used in pdep; port Wigner + Eckart? check
     what RMG-Py exposes and keep Wigner at minimum).
3. `rmgpu/data/thermo.py` (thin): the thermo MODEL side (not estimation): Wilhoit
   (port RMG-Py/rmgpy/thermo/wilhoit.pyx - Hf298/S298/Cp(T) polynomial) and NASA7
   (port nasa.pyx or delegate to Cantera's NASA9/7 + a thin numpy eval - decision:
   port Wilhoit in numpy (it is small and RMG-specific), use Cantera for NASA
   coefficient evaluation where convenient). get_heat_capacity, get_enthalpy,
   get_entropy, get_cp - all T-dependent, SI units.
   DO NOT port: group additivity, HBI, any "estimation" - job 04 (ML) is the only
   estimator (PLAN.md section 3).
4. `rmgpu/data/kinetics.py` (thin): retrieval logic for kinetics: given a reaction,
   check libraries (substructure match), else ML (job 04 interface - stub with a
   clear NotImplemented + TODO(job-04) for now). Libraries override ML.
5. `rmgpu/io/chemkin.py`: READ-only Chemkin parser (input for seeding/interoperability
   tests) is NOT required this job - skip (job 08/10). Just a stub note.

## Gate (job 02) -> gates/gate_02.py, report reports/job-02.md

  1. Round-trip count: for each library in a fixed list (use
     recommended_libraries.yml 'default' set + primaryThermoLibrary + training
     depository), entry count via rmgdb == entry count via RMG-Py loading the same
     library (run a small RMG-Py script to get counts; save baselines to
     gates/baselines/db_counts.json). Must be EXACT.
  2. Content hash: for primaryThermoLibrary + one kinetics library, dump all entries
     (label/sorted-smiles/coeffs/T-bounds) to sorted YAML from BOTH rmgdb (via rmgpu)
     and RMG-Py, and diff. Must be byte-identical after normalization (this catches
     data corruption in rmgdb).
  3. Lookup parity: for 25 random species from the superminimal/c3h4 examples,
     thermo lookup via rmgpu == thermo lookup via RMG-Py (same model type, same
     coefficients, tolerance 1e-12).
  4. Rate-model round-trip: for 100 reactions sampled from kinetics.db, construct
     the rmgpu rate model from the stored parameters and evaluate k(300,1bar) and
     k(1000,1bar); compare against RMG-Py's kinetics evaluation for the same stored
     parameters (script in gates/). Tolerance 1e-10 relative.
  5. Document any rmgdb coverage gaps found (in the report; add to STATUS.md
     decisions log if they affect later jobs).

## When done

STATUS.md update + commit "job-02: rmgdb database layer + round-trip gates" + STOP.

## Pitfalls

- rmgdb is a moving target owned by the user. If rmgdb has a bug or gap, DO NOT
  fork it into rmgpu; record the gap and query the raw RMG-database file for just
  that bit (document it in the report + STATUS.md), and flag it for the user.
- The depository ('training') is special: it feeds rate-rule TRAINING (a deleted
  feature) - for rmgpu it is just a kinetics library of curated data that ML models
  may be validated against. Treat it as a library.
- Family definitions (recipes/templates) are code (family.py files) - how rmgdb
  stores families must be checked empirically (query kinetics.db schema). If
  families are NOT in rmgdb, document it: recipes will be loaded from the family.py
  files via a controlled parse in job 05 (family.py files are data-ish Python;
  decide the loading strategy there and record it here as a note).
