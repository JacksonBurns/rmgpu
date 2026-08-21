# job-08: pdep MSC/RS/SLS + isotope + observables + diff/merge + interop exports

Read ORIENTATION.md and PLAN.md sections 4, 9 (plugin hooks are NOT this job),
10 (Phase 4 prep), 12. Prereqs: job-07 (CSE works + gated).

## Goal

Complete the pressure-dependence method set (MSC, RS, SLS) and the analysis/
interop tooling the parity suite (job 10) needs: isotope support, observables,
model diff/merge, and the legacy-format exports (Chemkin read, Cantera YAML,
RMS) from the canonical artifact.

## Deliverables

1. `rmgpu/pdep/msc.py`, `rs.py`, `sls.py` - port the other three lumping methods
   (RMG-Py/rmgpy/pdep/msc.pyx, rs.pyx, sls.py). Driver dispatches on the YAML
   pressure_dependence.method (cse|masc|rs|sls). Note: RMG's "masc"/"msc" naming -
   match RMG-Py's accepted method strings exactly.
2. `rmgpu/tools/isotopes.py` - isotope handling: isotope-aware species/thermo
   (RMG-Py/rmgpy/tools/isotopes.py, 39k) - the parity suite includes isotope
   examples; port the mass/isotopic-species logic + the thermo adjustments
   (zero-point energy shifts per isotope - port RMG's approach, no QM: RMG uses
   reduced-mass/symmetry corrections from the statmech layer).
3. `rmgpu/tools/observables.py` - observables + regression (RMG-Py/rmgpy/tools/
   observablesregression.py, 24k): the observable definitions (species
   conversion, time-to-x, profile integrals) and the regression comparison used
   by the parity gate. This is what job 10's gate will call.
4. `rmgpu/tools/diffmodels.py` + `mergemodels.py` - port from RMG-Py/rmgpy/tools/
   (24k, 8k): diff two mechanism artifacts (species/reaction sets, rate diffs),
   merge two. Operates on the canonical artifact (core.yaml) - this is why job 06
   made it loadable.
5. `rmgpu/io/canteramodel.py` - Cantera YAML export from the canonical artifact
   (RMG-Py/rmgpy/tools/canteramodel.py, 48k - but we export ONE route, PLAN.md
   12.3: `mechanism/cantera/chem.yaml`). Use the Cantera Python API (create the
   mechanism object, serialize) - do NOT hand-write the YAML.
6. `rmgpu/io/ckcsvparser.py` - Chemkin READER (for seeding/interop; port from
   RMG-Py/rmgpy/tools/ckcsvparser.py, 12k) so `rmgpu` can ingest a Chemkin
   mechanism as a seed (round-trip with the job-06 writer).
7. `rmgpu/export` CLI: `rmgpu export <run>/mechanism/core.yaml --to chemkin |
   cantera | rms` (RMS: YAML for ReactionMechanismSimulator - port the writer
   from RMG-Py's RMS export; it is small).

## Gate (job 08) -> gates/gate_08.py, report reports/job-08.md

  1. MSC/RS/SLS parity: re-run propane_branching (job 07's setup) with each of
     the 4 methods in rmgpu vs RMG-Py: per-reaction k(T,P) max relative diff +
     network-structure match. Same tolerances as job 07.
  2. Isotope: run an isotope example from RMG-Py's test/regression (pick one with
     isotopes - inventory them first) in rmgpu vs RMG-Py: species/thermo parity
     (ZPE shifts) + mechanism set parity.
  3. Observables: on the c3h4 run (job 06), compute the RMG-Py reference
     observables and compare to rmgpu's: max abs/rel diff per observable type.
  4. diff/merge: take two job-06 artifacts (superminimal at iter N and N+1):
     diff output matches RMG-Py's diffmodels on the equivalent models; merge
     round-trips (merge(A,B) contains both, no dupes).
  5. Export round-trips: core.yaml -> chemkin.inp -> re-read (ckcsvparser) ->
     core.yaml: species/reaction sets equal, rates match 1e-8. core.yaml ->
     cantera/chem.yaml: load in Cantera, check species/reaction counts + a sample
     of NASA coefficients + a sample of Falloff params vs the artifact.

## When done

STATUS.md + commit "job-08: pdep methods + isotope + observables + diff/merge +
exports" + STOP.

## Pitfalls

- The 4 pdep methods share the network/DoS core (job 07) - only the collision
  lumping differs; do not duplicate the network code.
- Cantera YAML export: Cantera's API is fussy about NASA coefficient ranges and
  Falloff representation (Lindemann vs Troe vs chemical-master-equation). RMG's
  canteramodel.py (48k) encodes the exact mapping - port its mapping logic, not
  its hand-written YAML.
- Isotope thermo: RMG's corrections are empirical/statistical (not QM) - verify
  against RMG-Py's numbers, do not derive your own.
