# Job & step prompts (execute in step order)

Two levels:

  prompts/job-NN-*.md          One file per JOB: a brief (goal, the step list,
                               the job's gate definition). NOT a task - a job is
                               too big for one session.
  prompts/steps/job-NN-step-MM-*.md  One file per STEP: a self-contained task
                               sized for ONE fresh human-started session. This is
                               what a session actually reads and executes.

## The jobs

    job-00-env-skeleton.md        env `rmgpu`, package skeleton, test scaffolding
    job-01-molecule.md            units + RDKit molecule layer + adjlist + atom types
    job-02-database.md            rmgdb-backed database layer + round-trip gates
    job-03-input-schema.md        YAML input schema + CLI + legacy .py importer
    job-04-ml-estimators.md       ML thermo/kinetics estimators + rate registry
                                  (the PoC thesis test lives in this job's gate;
                                  estimators are built FRESH per chemprop_example -
                                  RMG's ml/estimator.py is replaced, not reused)
    job-05-recipes.md             reaction recipe DSL + product enumeration
    job-06-coreloop-reactor.md    core/edge loop + torchdae reactor (first run)
    job-07-pdep-cse.md            statmech + master equation (CSE) + pdep parity
    job-08-pdep-methods-tools.md  pdep MSC/RS/SLS + isotope + observables +
                                  diff/merge + Chemkin/Cantera/RMS exports
    job-09-sensitivity.md         sensitivity/uncertainty via torchdae adjoint
    job-10-gas-parity.md          full gas-phase regression battery (parity milestone)
    job-11-plugin-solvation.md    plugin protocol + solvation plugin + liquid reactors
    job-12-catalysis.md           catalysis plugin (post-parity; design-validated)

Sequencing: steps within a job are strictly sequential (a step's output is the
next step's input). Jobs are sequential by their gates: do not start job N's
steps until job N-1's gate is GREEN and recorded in ../STATUS.md.
Job 10 is a campaign (its steps are battery chunks; the gate re-runs each
session until the battery is green). 11 needs 10; 12 needs 11.

## How a step session starts

The human reads ../STATUS.md's NEXT pointer, which names exactly ONE step file.
The human starts a fresh session whose task is: "Read prompts/steps/<that file>
and do it." The step file is self-contained: it repeats the job context it needs,
names the reference files to read (with a context budget), the deliverables,
the checks, and the done protocol. The session does ONLY that step, then stops.

Starting a session manually (the human's first message to the session):
  "Read STATUS.md (NEXT pointer) and execute exactly that next step.
   When done, update STATUS.md and commit. Do not start any other step.
   Do not spawn subagents."

## Step file anatomy

    # job-NN/step-MM: <title>
    Job context (job goal, prereq, the job gate)
    Context (invariant: repos, interpreter, GPU rules, no-fallbacks,
             the Chemprop-estimator replacement note)
    Where you are (prev/this/next step)
    Goal
    Reference to read (this step's budget)   <- the context budget: the listed
                                               reads must fit one session
    Deliverables
    Checks (must run and pass)
    Pitfalls
    Done protocol (commit, STATUS.md row + log entry, report, STOP)

If a step's reads turn out bigger than one session can hold, that is a
framework bug - the session records it in its report and stops; the
human splits the step (write the half-step files, update the step
table in ../STATUS.md, point NEXT at the first half).

## Report + commit conventions

  - Commit prefix: `job-NN/step-MM: <summary>` (no pushes).
  - Step report: reports/job-NN-step-MM-*.md (the checks' real output, not
    a paraphrase).
  - Job report: reports/job-NN.md (written by the gate step).
  - STATUS.md: the step row flips to `done` by the step session; the NEXT
    pointer moves by the human.
