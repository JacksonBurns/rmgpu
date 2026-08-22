# rmgpu -- RMG-GPU implementation repo

Ground-up, pure-Python (NumPy/PyTorch) rewrite of RMG. This repo is the working
repository: the plan, the job prompts that drive implementation across sessions,
and (from job 00 on) the code itself.

## Layout

    PLAN.md            The full feasibility + feature-parity plan (read the sections
                       your job prompt names; it is the source of truth for design).
    ORIENTATION.md     Self-contained context brief for implementation sessions.
                       EVERY job starts by reading this.
    STATUS.md          Work tracker: job table + append-only session log.
    prompts/           One file per job, in sequence. job-NN-<name>.md
    rmgpu/             (created in job 00) the package itself
    gates/             (created in job 00) parity-gate scripts + baselines
    reports/           (created in job 00) gate reports written by jobs

Reference material (read-only, NOT part of this repo, at /home/jackson/rmgpu):
  - RMG-Py/          reference implementation (v4.0.0-5-gd08392ed), conda env rmg_env has it installed
  - RMG-database/    the 2.4M-line data (families, libraries, statmech, transport), access via RMG-Py and its corresponding rmg_env
  - rmgdb/           SQL wrapper over RMG-database (our database layer), installed in rmgdb conda environment
  - chemprop_example/ example code for running inference with a trained Chemprop v2 model, use chemprop-dev environment if needed

## How work gets done (read this before starting a job)

Each job is a self-contained unit of work with its own deliverables and a testable
gate. The prompts are written so an agent with no other context can pick one up and
start coding: each prompt tells you exactly what to read, what to build, what the
acceptance criteria are, and what to do when done (update STATUS.md, commit, stop).

### Session vs subagent

Workflow: **one fresh subagent per job**, driven by the parent agent in the starting session.
The human will start you, an agent in a new session in this repo (cwd /home/jackson/rmgpu/rmgpu).
The agent (you) will then spawn off subagents in sequence according to this loop:

  1. First message: "Read ORIENTATION.md, then prompts/job-NN-*.md, then STATUS.md.
     Execute the job. When done, update STATUS.md and commit."  -- DO NOT interrupt the
     agent once this starts, let it keep working until it returns
  2. When the job finishes, review the commit + the STATUS.md entry + the gate report.
     Approve, or start a follow-up session to fix issues.
  3. Return to Step 1. spawning a new agent to work on the next job.

This is why we must do it this way:

- Each job is large (10k+ LOC of context of reference code). A fresh context window
  per job avoids compaction mid-job, which corrupts long implementation work.
- Jobs are strictly sequential with hard gates; there is nothing to parallelize
  BETWEEN jobs (a job's output is the next job's input), so subagent fan-out buys
  nothing at the job level.
- A new session is trivially resumable: the prompt file + STATUS.md is the whole state.

A subagent's self-report is not proof. Anything that writes files or runs gates
must be verified by the parent agent from the commit/diff.

Important note: similar to how *you* will only run one subagent at a time for each task (because
every execution on this machine happens on the same GPU, so it is impossible to run
multiple agents in parallel), it is **imperative** that your subagents do not
themselves spawn subagents. You should advise them of this in your prompts to them.

Also note that you DO NOT yet have access to actual trained Chemprop/CheMeleon models that are suitable for integration.
Your subagents should mark which ones are needed, but leave method stubs/signatures/etc. that just need the checkpoint dropped in.

### Discipline

- Do not start job N until job N-1's gate is GREEN and recorded in STATUS.md.
  Exceptions (skipping/deferring) require an explicit entry in STATUS.md.
- Gates are scripts in gates/ that compare rmgpu output against RMG-Py reference
  output (run on the same machine). Gate reports go to reports/ with real numbers.
  Never fabricate or "smooth" a gate result; a red gate is a finding, and it gets
  recorded as such.
- Commits: one or more per job, message prefix "job-NN: <summary>". No pushes.
- Conda env: `rmgpu` (created in job 00). Never install into system Python.

## Roadmap (see PLAN.md section 10 for the full version)

    00  env + package skeleton + test scaffolding
    01  units + molecule layer (RDKit wrapper, adjlist, atom types, resonance)
    02  database layer via rmgdb + round-trip hash gate
    03  YAML input schema + CLI + legacy .py importer
    04  ML estimators (CheMeleon thermo, Chemprop kinetics) + rate registry
    05  reaction recipe DSL + product enumeration (port from family.py)
    06  core/edge loop + torchdae reactor (first integration; superminimal/c3h4)
    07  statmech + master equation (CSE) + pdep gate (propane_branching)
    08  pdep MSC/RS/SLS + interpolation + isotope + observables/diff/merge
    09  sensitivity/uncertainty via torchdae adjoint
    10  full gas-phase parity (regression suite)  <-- the parity milestone
    11  plugin protocol + solvation plugin + liquid reactors
    12  catalysis plugin (post-parity; not required for core parity)
