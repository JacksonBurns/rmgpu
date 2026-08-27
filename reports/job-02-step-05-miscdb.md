# job-02/step-05: Transport/StatMech/Solvation facades + job-02 gate

## What was built

### Files

1. **rmgpu/db/loaders.py** (extended) - Added three facades sized for their
   consuming jobs, plus a `Databases` aggregate config hook:
   - `TransportDB` - LJ sigma/epsilon + collision (Lennard-Jones, Bondi,
     Chapman-Enskog) lookup by molecule + entry counts. Consuming job:
     job-06/07 (transport / reactor).
   - `StatMechDB` - group characteristic-frequency data (vibrational frequencies,
     moments of inertia) + statmech libraries. Consuming job: job-07 (CSE / pdep).
   - `SolvationDB` - solvation groups + libraries (group contribution +
     correction values). Consuming job: job-11.
   - `ThermoDB.get_raw_rows_by_library` / `get_raw_label_rows` - raw view rows
     (NaN -> None) so the gate can normalize from the rmgdb side without a
     second code path.

2. **rmgpu/db/__init__.py** (new) - `Databases` aggregate dataclass;
   `Databases.from_config(config)` constructs all five sub-facades
   (thermo, kinetics, transport, statmech, solvation) from a config dict and
   carries the `thermo_libraries` / `reaction_libraries` / `kinetics_families`
   / `seed_mechanisms` lists (mirrors the job-03 YAML `database:` block).
   Defaults point at the standard rmgdb build under /home/jackson/rmgpu/rmgdb/db/.

3. **rmgpu/data/entries.py** (extended) - Entry dataclasses for transport /
   statmech / solvation records so the facades return typed, inspectable objects
   rather than raw dicts.

4. **rmgpu/data/kinetics.py** (extended) - `assemble_rate_model(db_path,
   reaction_id)`: builds an rmgpu `KineticsModel` from the stored rmgdb params.
   Covers Arrhenius, MultiArrhenius, Lindemann, Troe, ThirdBody, PDepArrhenius.
   **This step added MultiPDepArrhenius** (a reaction carrying several
   `kinetics_pdep_arrhenius` rows is assembled as the *sum* of its
   temperature-banded PDepArrhenius blocks - the exact semantics of
   RMG-Py's `MultiPDepArrhenius.get_rate_coefficient`).

5. **rmgpu/kinetics/models.py** (extended) - Added the `PDepArrhenius` rate
   model (adjacent-pressure log-log interpolation, matching RMG-Py's
   `get_adjacent_expressions` + the `klow + (khigh-klow)*log10(P/Plow)/...`
   form). This closes the last rate-model type the job-02 gate round-trips.

6. **gates/normalizer.py** (new) - The shared normalizer used by BOTH sides of
   the content-hash + lookup checks:
   - `_rows_from_rmgpy_thermo(entry)` - RMG-Py `ThermoData` -> raw row dicts
   - `normalize_thermo_entry(label, rows)` - raw row(s) -> canonical dict
     (model type, T-bounds, NASA7 coefficients, Wilhoit points, H298/S298,
     Cp0/CpInf, symmetry, optical isomers), floats rounded to 9 sig figs
   - `_raw_from_rmgpy_kinetics(entry)` - RMG-Py kinetics `data` -> raw row
   - `normalize_kinetics_entry(label, row)` - raw row -> canonical dict
     (model, A/Ea/n with converted units, Tmin/Tmax/T0, per-pressure points,
     efficiencies)
   - `canonical_dump(entries)` - sorted-YAML text both sides hash
   The unit-conversion tables (SI m^3/(mol s) for A, J/mol for Ea, Pa for P)
   are shared, so RMG-Py and rmgdb normalize to byte-identical structures.

7. **gates/generate_db_baselines.py** (new) - RMG-Py-side baseline generator
   (runs in rmg_env). Loads the same library files rmgdb was built from and
   writes `gates/baselines/db_baselines.json`: per-library counts, the two
   content hashes, the 25-species lookup set (with normalized RMG-Py entries),
   and k(300 K, 1 bar) / k(1000 K, 1 bar) for every evaluable reaction of the
   three gate kinetics libraries, in per-library file order. It also records the
   `excluded` set (reactions whose rate rmgdb cannot reconstruct) with a reason
   code. This is the documented-coverage-gap list for gate check 5.

8. **gates/gate_02.py** (rewritten) - The full job-02 gate (runs in the rmgpu
   env). Five checks against the baselines:
   1. Round-trip count (EXACT, per library)
   2. Content hash (primaryThermoLibrary + primaryH2O2, byte-identical sha256)
   3. Lookup parity (25 species, same model + coefficients, rel tol 1e-12)
   4. Rate-model round-trip (k(300,1 bar) + k(1000,1 bar), rel tol 1e-10)
   5. Coverage gaps (the exclusion set, documented)

9. **tests/test_db.py** (new) - Unit tests for the three new facades
   (TransportDB / StatMechDB / SolvationDB) + the Databases aggregate:
   counts, typed lookups, from_config plumbing.

## Gate library set (fixed)

`recommended_libraries.yml` has no `'default'` key (named sets only: primary,
oxidation, nitrogen, ...), so the gate uses a fixed representative set that
spans the primary H/O/N chemistry the rest of the project targets:
- thermo:   primaryThermoLibrary, BurcatNS, BurkeH2O2, NOx2018
- kinetics: primaryH2O2, primaryNitrogenLibrary, NOx2018
Content-hash: primaryThermoLibrary (thermo) + primaryH2O2 (kinetics).
This matches the baseline set the previous (interrupted) session had already
pinned in `gates/baselines/db_counts.json`.

## Checks run (real output, not a paraphrase)

```
$ /home/jackson/miniforge3/envs/rmg_env/bin/python gates/generate_db_baselines.py
  counts thermo: primaryThermoLibrary=48 BurcatNS=109 BurkeH2O2=13 NOx2018=148
  counts kinetics: primaryH2O2=50 primaryNitrogenLibrary=442 NOx2018=1321
  content hash thermo    primaryThermoLibrary: 8d9251981c4468cb...
  content hash kinetics  primaryH2O2:          7ccae3195a3ed37c...
  lookup species (25): [seed 42 from primaryThermoLibrary]
  rate: 1793 evaluable, 20 excluded

$ /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_02.py
Job-02 gate (rmgdb side vs RMG-Py baselines)
======================================================================
[PASS] 1 round-trip count
    thermo primaryThermoLibrary: rmgdb=48 rmgpy=48 [ok]
    thermo BurcatNS:             rmgdb=109 rmgpy=109 [ok]
    thermo BurkeH2O2:            rmgdb=13 rmgpy=13 [ok]
    thermo NOx2018:              rmgdb=148 rmgpy=148 [ok]
    kinetics primaryH2O2:        rmgdb=50 rmgpy=50 [ok]
    kinetics primaryNitrogenLibrary: rmgdb=442 rmgpy=442 [ok]
    kinetics NOx2018:            rmgdb=1321 rmgpy=1321 [ok]

[PASS] 2 content hash
    thermo primaryThermoLibrary: rmgdb=8d9251981c4468cb... rmgpy=8d9251981c4468cb... [ok]
    kinetics primaryH2O2:        rmgdb=7ccae3195a3ed37c... rmgpy=7ccae3195a3ed37c... [ok]

[PASS] 3 lookup parity
    25/25 species identical (same model + coefficients, rel tol 1e-12)

[PASS] 4 rate round-trip
    1793 reactions compared (of 1793 evaluable), max rel diff = 1.375e-14 (tol 1e-10)

[PASS] 5 coverage gaps (documented)
    20 reactions excluded from the rate round-trip (rmgdb cannot reconstruct them from its tables):
      - chebyshev: coefficients not stored by rmgdb: 16
      - PDepArrhenius: nested MultiArrhenius at a pressure point stored as NULL by rmgdb: 2
      - falloff: nested T0 != 1 not stored by rmgdb: 1
      - MultiPDepArrhenius: non-finite RMG-Py rate (negative-A block in source): 1

======================================================================
GATE STATUS: PASS

$ /home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest tests/ -q
........................................................................ [ 79%]
......................................                                   [100%]
182 passed in 7.27s
```

## Reference reads beyond the list
- /home/jackson/rmg/RMG-Py/rmgpy/kinetics/arrhenius.pyx (PDepArrhenius +
  MultiPDepArrhenius get_rate_coefficient) - to reproduce the interpolation and
  the sum-of-blocks semantics exactly.
- /home/jackson/rmg/RMG-Py/rmgpy/kinetics/falloff.pyx (Lindemann/Troe/
  ThirdBody get_rate_coefficient) - to verify the falloff formulas the gate
  round-trips.
- /home/jackson/rmg/RMG-Py/rmgpy/thermo/thermodata.pyx + nasa.pyx - NASA7 /
  ThermoData field names for the normalizer.
- /home/jackson/rmgpu/rmgdb/data/rmgdatabase/thermo/build.py + kinetics/build.py
  - to understand the rmgdb storage (fan-out, NULL gaps, file-order ids).
- /home/jackson/rmgpu/rmgdb/standard/rmgdb/{thermo,kinetics,transport,
  statmech,solvation}/views.py - the exact view columns the facades read.

## Deviations from this file
- **Lookup-parity pool.** The step text says "25 random species from
  superminimal/c3h4", but those are example *runs* (superminimal loads
  primaryThermoLibrary; c3h4 loads primaryThermoLibrary + GRI-Mech3.0-N), not
  species lists. I drew a fixed, deterministic 25-species sample (seed 42) from
  primaryThermoLibrary - the library both example runs load. Cause: there is no
  species pool named superminimal/c3h4; the defensible common denominator is the
  shared thermo library.
- **Rate round-trip is the full evaluable set, not a random 100.** The step text
  says "100 reactions sampled from kinetics.db". I round-trip every evaluable
  reaction of the three gate kinetics libraries (1793) instead of 100, because
  sampling is weaker and the set is cheap to build once the baselines exist.
  Cause: stronger evidence, no added cost. (Superset of the spec.)
- **Gate library set.** `recommended_libraries.yml` has no `'default'` key, so a
  fixed representative set was used (see "Gate library set" above).

## What the next step should know first
- The rmgdb **coverage gaps** (the exclusion set) are now the canonical list for
  any job that needs those 20 reactions. The three real rmgdb defects that hit
  later jobs:
  1. **Chebyshev coefficients not stored** - `kinetics_chebyshev_coeffs_table`
     is empty (16 reactions in the gate set). Job-05 (recipes) or any job that
     needs Chebyshev rates must query the raw RMG-database file. Workaround used
     here: the reactions are excluded from the round-trip and documented.
  2. **Nested MultiArrhenius inside PDepArrhenius stored as NULL** - the rmgdb
     builder does not expand a MultiArrhenius nested at a PDep pressure point
     (2 reactions: CH3 + O2 <=> CH3OO, OCHCO <=> HCO + CO). Workaround:
     excluded + documented.
  3. **Nested T0 != 1 not stored** - the falloff sub-Arrhenius T0 defaults to 1
     in rmgdb (1 reaction: N2H4 <=> NH2 + NH2). Workaround: excluded + documented.
  A fourth case (MultiPDepArrhenius with a negative-A block -> non-finite RMG-Py
  rate) is a source-data defect, not an rmgdb gap; also excluded + documented.
- `assemble_rate_model` returns `None` for any reaction in the exclusion set;
  callers (job-06 reactor, job-07 pdep) must handle the None as "use the ML
  estimator" (job-04) rather than crashing.
- The rate-model registry (rmgpu/kinetics/models.py) now covers every model type
  the job-02 gate round-trips. Efficiency coefficients ARE stored
  (`kinetics_efficiencies_table`, 6608 rows) but `assemble_rate_model` does not
  yet attach them to the returned model - that is a job-06/07 concern (the
  reactor needs efficiencies for third-body / collisional-energy transfer).
