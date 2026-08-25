# job-08: pdep MSC/RS/SLS + isotope + observables + diff/merge + exports

Status of this file: a JOB BRIEF, not a task. Do not try to "do this job"
in one session - it is decomposed into the steps below, one session each.
The human reads STATUS.md's NEXT pointer; this file is the map of the job.

## Goal

Complete the pressure-dependence method set (MSC, RS, SLS - onto job 07's network/DoS core) and the analysis/interop tooling the parity suite (job 10) needs: isotope support, observables, model diff/merge, and the legacy-format exports (Chemkin read, Cantera YAML, RMS) from the canonical artifact.

## Prereq

job-07 done (CSE works + gated; the network/DoS core)

## Steps (strictly sequential; one fresh human-started session each)

  step 01  prompts/steps/job-08-step-01-methods.md  pdep MSC + RS + SLS (onto job-07's core)
  step 02  prompts/steps/job-08-step-02-isotopes.md  Isotope support
  step 03  prompts/steps/job-08-step-03-observables.md  Observables + regression comparison
  step 04  prompts/steps/job-08-step-04-diffmerge.md  diffmodels + mergemodels (on the canonical artifact)
  step 05  prompts/steps/job-08-step-05-exports.md  Cantera export + Chemkin reader + the export CLI
  step 06  prompts/steps/job-08-step-06-gate.md  Job-08 gate (methods + isotope + observables + exports)

## The job gate

Run by the final step's session (gates/gate_08.py, report
reports/job-08.md):

gates/gate_08.py: 1. MSC/RS/SLS parity: propane_branching (job 07's setup) with each of the 4 methods in rmgpu vs RMG-Py: per-reaction k(T,P) max relative diff + network-structure match (same tolerances as job 07). 2. Isotope: an isotope example from test/regression (inventory first) in rmgpu vs RMG-Py: species/thermo parity (the ZPE shifts) + mechanism-set parity. 3. Observables: on the c3h4 run (job 06), RMG-Py's reference observables vs rmgpu's: max abs/rel diff per observable type. 4. diff/merge: two job-06 artifacts (superminimal at iter N and N+1): the diff matches RMG-Py's diffmodels on the equivalent models; the merge round-trips (merge(A,B) contains both, no dupes). 5. Export round-trips: core.yaml -> chemkin.inp -> re-read (ckcsvparser) -> core.yaml: species/reaction sets equal, rates match 1e-8. core.yaml -> cantera/chem.yaml: load in Cantera, check species/reaction counts + a sample of NASA coeffs + a sample of Falloff params vs the artifact.

## When the job is done

The final step's report (reports/job-08.md) has the gate result, the
job table row is `done` (or `blocked` with the cause), and the NEXT pointer
in STATUS.md targets job-09's first step (if there is a next job).
