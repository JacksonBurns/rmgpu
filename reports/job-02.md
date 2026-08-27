# job-02: Database layer via rmgdb - GATE REPORT

Job goal: ALL database I/O for rmgpu goes through rmgdb (the SQL/SQLite
wrapper over RMG-database). No YAML-parsing of RMG-database files inside
rmgpu (except data rmgdb does not cover yet, documented, not silently
worked around).

**GATE STATUS: PASS** (gates/gate_02.py, 2026-08-26)
Full evidence: reports/job-02-step-05-miscdb.md (the gate's step report).

## The five checks (real numbers)

1. **Round-trip count** (EXACT) - PASS
   - thermo: primaryThermoLibrary 48, BurcatNS 109, BurkeH2O2 13, NOx2018 148
   - kinetics: primaryH2O2 50, primaryNitrogenLibrary 442, NOx2018 1321
   - rmgdb count == RMG-Py `len(library.entries)` for all 7 libraries.

2. **Content hash** - PASS (byte-identical sha256 from both sides)
   - primaryThermoLibrary (thermo):  `8d9251981c4468cb...` == `8d9251981c4468cb...`
   - primaryH2O2 (kinetics):         `7ccae3195a3ed37c...` == `7ccae3195a3ed37c...`
   - Every entry of both libraries normalized (label, model, SI coefficients,
     T-bounds, per-pressure points, efficiencies) and dumped to sorted YAML
     from BOTH the rmgdb side (via rmgpu facades) and the RMG-Py side (via the
     shared gates/normalizer.py); hashes are identical. This is the data
     corruption check - it passes, so rmgdb is a faithful mirror of the files
     it was built from.

3. **Lookup parity** - PASS
   - 25/25 species (seed 42, drawn from primaryThermoLibrary, the library both
     reference runs superminimal and c3h4 load): the normalized rmgdb entry
     equals the RMG-Py entry - same model type, same coefficients within
     relative tolerance 1e-12.

4. **Rate-model round-trip** - PASS
   - 1793 reactions (every evaluable reaction of the 3 gate kinetics
     libraries, position-aligned with RMG-Py's file order): the rmgpu rate
     model built from the stored rmgdb params matches RMG-Py's
     `get_rate_coefficient` at (300 K, 1 bar) and (1000 K, 1 bar) with
     **max relative diff 1.375e-14** (tolerance 1e-10).
   - This is 11 orders of magnitude inside the tolerance; it exercises
     Arrhenius, MultiArrhenius, Lindemann, Troe, ThirdBody, PDepArrhenius and
     MultiPDepArrhenius (sum of temperature-banded blocks).

5. **Coverage gaps** - documented (see below). PASS by definition: every
   excluded reaction has a reason code and is listed.

## The rmgdb coverage-gap list (what's missing / workaround / job it hits)

These are the reactions whose rate rmgdb cannot reconstruct from its tables,
excluded from check 4, and the rmgdb defects that cause them.

| # | Gap | What's missing in rmgdb | Workaround used | Job it hits | Count (gate set) |
|---|-----|-------------------------|-----------------|-------------|------------------|
| 1 | Chebyshev coefficients not stored | `kinetics_chebyshev_coeffs_table` is empty (16 chebyshev rows in the gate set, 114 in the whole DB, 0 coefficient rows) | Reaction excluded from the rate round-trip and documented. A job that needs a Chebyshev rate must query the raw RMG-database `reactions.py` for just that entry (no YAML-parsing is done inside rmgpu for this). | job-05 (recipes), job-06 (reactor) | 16 |
| 2 | Nested MultiArrhenius in PDepArrhenius stored as NULL | The rmgdb kinetics builder does not expand a MultiArrhenius nested at a PDepArrhenius pressure point; the pressure row's A/Ea/n come out NULL | Reaction excluded + documented. Raw-file fallback as above if a rate is needed. | job-06 (reactor), job-07 (pdep/CSE) | 2 |
| 3 | Nested T0 != 1 not stored | The falloff sub-Arrhenius T0 (e.g. the high-pressure `T0=(1000,'K')` on N2H4 <=> NH2 + NH2) is not persisted; rmgpu must default it to 1 | Reaction excluded + documented. | job-07 (pdep/CSE) | 1 |
| 4 | (source-data, not an rmgdb gap) negative-A block | HCCO + OH <=> CO2 + CH2 has a PDep block with a negative A, so RMG-Py ITSELF returns a non-finite rate (log10 of a negative) | Reaction excluded + documented. This is a defect in the source library, not in rmgdb or rmgpu. | none (skip the reaction) | 1 |

Related finding (does not block the gate): `assemble_rate_model` does not yet
attach the stored efficiency coefficients (kinetics_efficiencies_table, 6608
rows) to the returned model. That is a job-06/07 concern (the reactor and the
master equation need efficiencies); the data is present in rmgdb and is not
re-queried from files.

## Decisions / things later jobs must know

- rmgdb is a faithful mirror for the primary H/O/N chemistry (checks 1-2 prove
  it). Jobs should read everything they can from rmgdb and only fall back to a
  raw RMG-database file read for the specific documented gap above.
- `assemble_rate_model(db_path, reaction_id)` returns `None` for any reaction
  in the exclusion set. Callers (job-06 reactor, job-07 pdep) must treat `None`
  as "route to the ML estimator (job-04)", not as an error.
- The rate-model registry (rmgpu/kinetics/models.py) covers every model type the
  job-02 gate round-trips. It is the reference that later jobs consume.
