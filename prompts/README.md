# Job prompts (execute in order)

One file per job. Each is self-contained: it names the reference code to read,
the deliverables, the gate (a script in ../gates/ + a report in ../reports/), and
the "when done" protocol (update ../STATUS.md, commit, stop).

    job-00-env-skeleton.md        env `rmgpu`, package skeleton, test scaffolding
    job-01-molecule.md            units + RDKit molecule layer + adjlist + atom types
    job-02-database.md            rmgdb-backed database layer + round-trip gates
    job-03-input-schema.md        YAML input schema + CLI + legacy .py importer
    job-04-ml-estimators.md       ML thermo/kinetics estimators + rate registry
                                  (the PoC thesis test lives in this job's gate)
    job-05-recipes.md             reaction recipe DSL + product enumeration
    job-06-coreloop-reactor.md    core/edge loop + torchdae reactor (first run)
    job-07-pdep-cse.md            statmech + master equation (CSE) + pdep parity
    job-08-pdep-methods-tools.md  pdep MSC/RS/SLS + isotope + observables +
                                  diff/merge + Chemkin/Cantera/RMS exports
    job-09-sensitivity.md         sensitivity/uncertainty via torchdae adjoint
    job-10-gas-parity.md          full gas-phase regression battery (parity milestone)
    job-11-plugin-solvation.md    plugin protocol + solvation plugin + liquid reactors
    job-12-catalysis.md           catalysis plugin (post-parity; design-validated)

Sequencing: 00-07 are strictly sequential (each gate is the next job's input).
08-09 can start once 07 is green (09 depends on 06+04, 08 on 07). 10 is a campaign
job (may span many sessions). 11 needs 10 green; 12 needs 11.

Starting a session: read ../ORIENTATION.md, then the job file, then ../STATUS.md
(tail). Work the job. Update STATUS.md. Commit. Stop. See README.md for the
session-vs-subagent guidance.
