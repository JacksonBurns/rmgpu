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
  - RMG-Py/          reference implementation (v4.0.0-5-gd08392ed)
  - RMG-database/    the 2.4M-line data (families, libraries, statmech, transport)
  - rmgdb/           SQL wrapper over RMG-database (our database layer)

## How work gets done (read this before starting a job)

Each job is a self-contained unit of work with its own deliverables and a testable
gate. The prompts are written so an agent with no other context can pick one up and
start coding: each prompt tells you exactly what to read, what to build, what the
acceptance criteria are, and what to do when done (update STATUS.md, commit, stop).

### Session vs subagent: use a NEW SESSION per job (recommended)

Recommended workflow: **one fresh session per job**, driven by the user.

  1. Start a new session in this repo (cwd /home/jackson/rmgpu/rmgpu).
  2. First message: "Read ORIENTATION.md, then prompts/job-NN-*.md, then STATUS.md.
     Execute the job. When done, update STATUS.md and commit."
  3. When the job finishes, review the commit + the STATUS.md entry + the gate report.
     Approve, or start a follow-up session to fix issues.
  4. Say "next job" (or start the next session) to proceed.

Why a new session rather than subagents dispatched from one session:
- Each job is large (10k+ LOC of context of reference code). A fresh context window
  per job avoids compaction mid-job, which corrupts long implementation work.
- Jobs are strictly sequential with hard gates; there is little to parallelize
  BETWEEN jobs (a job's output is the next job's input), so subagent fan-out buys
  nothing at the job level.
- A new session is trivially resumable: the prompt file + STATUS.md is the whole state.

Within a single job, subagents MAY be used for parallelizable subtasks (e.g. porting
independent statmech rotors, or generating test fixtures from RMG-Py reference
outputs). Two hard rules for subagents here:
  1. Single GPU, one heavy GPU task at a time. Never run two GPU-hungry subagents
     concurrently; the local model server (llama-server) must not be killed.
  2. A subagent's self-report is not proof. Anything that writes files or runs gates
     must be verified by the parent (or the user) from the commit/diff.

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
