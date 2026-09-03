# job-06/step-07: Re-open the job-06 gate (it was closed GREEN on false evidence) - make the gate honest + make c3h4 actually run

Job: job-06 - Core/edge mechanism loop + torchdae reactor (first integration)
Prereq: job-06/step-06 (the gate) was marked done, but a follow-up audit found the
gate was GREEN on **false evidence**. This step re-opens it.
This step is a FIX step (like job-05/step-06), written to be executed IN ONE SHOT
by a fresh agent that has no other context. It names every bug with its exact
file/line, gives the exact fixes, and defines the HONEST gate that must turn
GREEN.

## Why this step exists (the false-GREEN)

The gate report `reports/job-06-step-06-gate.md` says, verbatim:
> "The superminimal parity matches the recorded reference (core identical, edge
> within tolerance). Deviations: None."

The gate's OWN recorded results (`reports/gate_06_results.json`, on disk) say the
opposite:
- `c3h4.status = FAIL` (`reason: "rmgpu run failed: Invalid SMILES string: InChI=1/N2/c1-2"`)
- `superminimal` core species **20** vs RMG-Py **13** (`identical: false`, 15 O-chain
  species rmgpu-only), core reactions **100** vs **19** (only 5 shared)
- `output_tree.resimulate_final.max_abs_diff = 213.846`, `within_tol: false`
  (the written final profile row contains mole fractions of ~250 and negatives -
  a non-physical blow-up)
- `all_hard_pass = true` (so it exited 0) because the hard-check set is
  `{subgate, run_completes, output_tree, provenance}` and **excludes** c3h4 and
  the resimulate-within-tolerance.

So the gate LIES: it reports GREEN while its own JSON records a FAIL and
non-physical profiles. That is the defect to fix. Two things are required:

1. **The gate must be honest** - its hard checks must catch a c3h4 that did not
   actually run a real mechanism, non-physical profiles, and hidden coverage.
   A run that "completes" by silently dropping the entire seed mechanism is a
   FAIL, not a PASS.
2. **c3h4 must actually run** - right now it runs with ZERO seed reactions
   (see bug B1). The seed-mechanism path is a stub.

### The parity bar (user direction, 2026-09-03 - read this before touching the
### gate's pass/fail logic)

> We do NOT need exact parity in the simulation results. They should be
> qualitatively similar, but deviations can be considered improvements in rmgpu
> where appropriate.

Consequences for this step:
- The gate must **NOT** require an identical core species/reaction set vs RMG-Py.
- Parity vs RMG-Py is a **recorded, qualitative finding**: compute the set diffs,
  list the systematic divergences, root-cause each to a module, and note whether
  the deviation is a known rmgpu behavior (e.g. more aggressive screening) that is
  acceptable or an improvement. A divergent core WITH a documented cause is a
  PASS, not a failure.
- What the gate MUST hard-fail on (the real stack-health criteria):
  - both examples actually run (c3h4 with a NON-EMPTY seed mechanism, not 3 species);
  - the written profiles are physically valid (every mole fraction in [0, 1],
    the mole-fraction sum per row within ~1e-3 of 1);
  - the output tree exists and core.yaml loads back;
  - provenance is real;
  - estimation/coverage counts are the REAL numbers from the run (not hardcoded {}).

Do not "fix" the O-chain over-generation to force an identical core. Document it.

## The bugs (all verified against the tree on 2026-09-03, exact locations)

**B1 - c3h4 never loads a seed mechanism (the big one).**
`examples/c3h4.yaml` lists `seed_mechanisms: [GRI-Mech3.0-N]`. That name does
NOT exist in rmgdb's kinetics library table. The real library is
`GRI-Mech3` (54 species, 307 reactions; 3 of them multi-band Arrhenius; 6
termolecular `cm^6/(mol^2*s)` rows). So `load_seed_mechanism('GRI-Mech3.0-N', ...)`
raises `ValueError: Seed mechanism 'GRI-Mech3.0-N' not found in rmgdb`. That
exception is caught in `rmgpu/main.py:130` by a bare
`print(f"warning: seed mechanism {name!r} load failed: {e}")` and **swallowed** -
the run proceeds with the 3 seed species (CH2/C2H2/N2) and ZERO seed reactions.
The c3h4 run therefore "completes" a trivial 3-species mechanism and the gate
marks it PASS. (The recorded FAIL was from an earlier run that crashed on the
InChI parse, before the InChI fix in main.py landed; after that fix the run no
longer crashes, so it now silently "passes" a stub.)
- Note the asymmetry: `GRI-Mech3.0-N` DOES exist in thermo.db (51 entries) but
  NOT in the kinetics library table. So the thermo name matches and the kinetics
  name doesn't. The loader only resolves the kinetics name.

**B2 - seed_loader placeholder-methane (silent garbage chemistry).**
`rmgpu/db/seed_loader.py:191`: when a reaction's species label is not in the
library dictionary, it builds
`Species(label=lbl, molecule=Molecule(smiles="C"))` - a **methane placeholder**.
This injects bogus methane into any reaction whose species is missing. For
`GRI-Mech3` specifically the dictionary covers all 307 reactions' species (0
missing), so this path is currently dormant - but it is a correctness landmine
and must be removed or made loud. (The 5 placeholder species seen in an earlier
probe were from loading a different library.)

**B3 - seed_loader discards thermo it computes.**
`rmgpu/db/seed_loader.py:171-174`: `thermo_entry = _load_thermo_for_label(...)`
is computed, then line 172 builds `Species(label=..., molecule=mol, reactive=True)`
WITHOUT attaching `thermo=thermo_entry`. So every seed species falls through to
ML estimation even when thermo.db has a library value. Wasted work + inconsistent
with the "library hit first" rule.

**B4 - seed_loader multi-band Arrhenius is `LIMIT 1`.**
`rmgpu/db/seed_loader.py:90-100` (`_load_arrhenius`) uses `LIMIT 1`, silently
picking one temperature band for multi-band reactions (GRI-Mech3 has 3). This
should not silently drop bands - either assemble the full model (job-02's
`assemble_rate_model` already handles MultiArrhenius) or record the gap.

**B5 - seed_loader termolecular A-unit heuristic is wrong.**
`rmgpu/db/seed_loader.py:107-108`: `if A_unit and "cm" in str(A_unit): A *= 1e-6`.
`cm^3/(mol*s)` -> m^3/(mol*s) is `*1e-6` (correct), but
`cm^6/(mol^2*s)` (termolecular, 6 rows in GRI-Mech3) needs `*1e-12`, not `*1e-6`
- a factor of 1e6 error on exactly the pressure-dependent termolecular reactions.
Convert by the ACTUAL unit, not a substring test.

**B6 - gate hard checks exclude the things that are actually broken.**
`gates/gate_06.py:584-589`: `hard = {subgate, run_completes, output_tree,
provenance}`. `run_completes` only checks `status in (PASS, MAX_ITER)`. c3h4's
check sets `out["status"]="PASS"` at line 358 unconditionally whenever the run
doesn't raise - and the run never raises because main.py swallows the seed
failure (B1). And `resimulate_final.within_tol` (line 534, tol 1e-3) is computed
but never enforced.

**B7 - gate fast-path hardcodes empty coverage.**
`gates/gate_06.py:228-238`: when the run output already exists (the normal
re-check path), the summary dict is built with `"estimation_counts": {}` and
`"coverage": {}` - the real numbers in `summary.md` (library 2 / ml 491 /
coverage_errors 550; thermo 402 / kinetics 148 dropped) are never parsed. So the
gate's JSON hides the coverage story.

**B8 - the divergence-cause block is stale and misattributed.**
`gates/gate_06.py:281-295` (`_divergence_cause`) describes "70 reactions, 11
O-chain species" and claims the over-generation is "addressed by job-07 (pdep on
in both sides)." The ACTUAL recorded run has **100** core reactions and **15**
O-chain species (it describes an earlier run). And pdep does not stop O-chain
diradical recombination enumeration or the screening promotion. The real root
cause (see Fix 6) is the screening criterion.

## The fixes (exact mechanism)

### Fix 1 - make the seed-mechanism name resolve to a real rmgdb kinetics library
`rmgpu/db/seed_loader.py`, `_library_id_by_name` (line 34).
The c3h4.yaml seed name `GRI-Mech3.0-N` is the RMG-Py library name; rmgdb's
kinetics library is `GRI-Mech3`. Do NOT edit c3h4.yaml to a different name (it is
a lossless import of RMG-Py's input.py - job-03's guarantee). Instead, make the
resolution robust and LAUD:
- Try the exact name first.
- On a miss, try a small **alias table** (module-level dict, e.g.
  `{"GRI-Mech3.0-N": "GRI-Mech3"}`) and/or a controlled normalization (strip a
  trailing `.N` / version suffix). Record which alias/normalization resolved.
- If still unresolved, **raise** (the existing `ValueError` is fine) - the point
  is that Fix 3 makes the caller NOT swallow it.
- Keep the thermo name separate (thermo.db does have `GRI-Mech3.0-N`).
Add a unit test: `load_seed_mechanism` on the c3h4 seed name returns a
non-empty species list AND a non-empty reaction list (assert both > 0).

### Fix 2 - kill the placeholder-methane; make missing species loud
`rmgpu/db/seed_loader.py:186-197`. Remove the `Molecule(smiles="C")` placeholder.
For a reaction species label not in the library dictionary: **skip that
reaction** and count it as a `seed_species_gap` (do NOT inject methane). GRI-Mech3
has 0 such gaps so this should be a no-op for c3h4, but the behavior must be
correct and loud. Return a small summary (e.g. `n_reactions_dropped_missing_species`)
alongside `(species_list, reaction_list)` OR attach it to the returned reaction
objects; record it in the run summary.

### Fix 3 - main.py must not swallow seed-mechanism failures
`rmgpu/main.py:92-135` (`_build_seed_mechanisms`). The `try/except` at line ~128-130
currently does `print(warning)` and returns empty. Change it so a seed-mechanism
load failure is a **recorded, visible error**:
- Do not return an empty list silently. Either (a) raise so `M.run` propagates it
  (c3h4 then FAILs loudly - which is the honest outcome until Fix 1 lands), or
  (b) record the failure in the run context so the run summary and the output
  tree carry `seed_mechanism_error: <name>: <msg>` and the gate sees it.
  Prefer (a) for correctness (a run that asked for a seed mechanism and got none
  is not a valid run) and confirm c3h4 loads the seed end-to-end.
- Whatever you pick, the gate (Fix 4) must be able to tell "seed was requested
  but not loaded" apart from "no seed requested."

### Fix 4 - the honest gate hard checks
`gates/gate_06.py`, `main()` (line 571) + `check_c3h4` (line 301).
Redo the pass/fail so the gate cannot lie:
- `check_c3h4`: the status is PASS only if ALL of:
  - the run completed (no exception) - AND -
  - the seed mechanism was actually loaded: assert the c3h4 core species count is
    well above the 3 seed species (the GRI seed is 54 species; assert
    `core_species_count >= 30` as a sanity floor, and record the actual number),
    AND the seed reaction count is non-zero;
  - the c3h4 output tree exists and `mechanism/core.yaml` loads back;
  - the c3h4 final profile is physically valid (Fix 5).
  If the seed was not loaded, `status = "FAIL"` with `reason` naming it, and the
  hard check fails.
- Add a `c3h4` hard check: `results["c3h4"]["status"] == "PASS"`.
- Keep the parity-vs-RMG-Py for c3h4 as a **recorded finding** (set diffs), NOT a
  hard gate. c3h4 parity is expected to diverge (different screening); record the
  counts + diffs + a one-line cause. Do not hard-fail on it.
- `run_completes` for superminimal stays (it runs fast and completes).

### Fix 5 - physical-validity check (new, both examples)
`gates/gate_06.py`. Add a check that reads the FINAL row of
`profiles/reactor/time_series.csv` (and, cheaply, every row) for BOTH the
superminimal and c3h4 output trees:
- every mole-fraction column value is in `[0.0, 1.0]` (tolerance: `0 <= v <= 1`
  with a small epsilon, e.g. `v >= -1e-6 and v <= 1 + 1e-6`);
- the row sum is within `1e-3` of 1.0 (the state is a closed mole-fraction
  vector).
If any value is out of range or the sum is off, `status = "FAIL"` with the
offending species/value recorded. This is what catches the 213.8 / ~250
non-physical blow-up. Add a `physical_validity` hard check (superminimal AND
c3h4).
NOTE: if the superminimal final profile is genuinely non-physical after your
fixes, that is a REAL finding (the loop's screen + no-demotion drives O-chain
diradicals to a blow-up). Record it precisely (which species, the value, the
iteration) and report it - do NOT loosen the check to make it pass, and do NOT
tune tolerances to force a "physical" result. A non-physical final profile is a
RED that the report must explain (see the "honest outcome" note below).

### Fix 6 - rewrite the divergence-cause block to match the REAL run
`gates/gate_06.py:281-295` (`_divergence_cause`). Replace the stale text with
what the ACTUAL recorded run shows (100 core reactions, 15 O-chain species) and
the correct root cause:
- rmgpu's `_screen` (`rmgpu/core/loop.py:430-465`) promotes a species to the
  core when the max of its reactions' forward/reverse rate-ratio exceeds
  `tolerance_move_to_core`, evaluated on a **fixed-timescale snapshot**
  (`t_end = CHAR_RATE_TFACTOR / char`, line 419, factor 5.0). It has **no
  rate-ratio demotion** - `prune()` (line 556-564) only removes edge species not
  referenced by an edge reaction.
- RMG-Py's screen (`rmgpy/rmg/model.py:1418-1455`) uses reactor-driven
  `max_edge_species_rate_ratios` (from the actual reactor solution, not a
  fixed snapshot) and prunes below `tolerance_keep_in_edge` AND keeps in the edge
  between keep and move-to-core.
- So the O-chain diradicals (H-O_n-H, O_n) get promoted and stay in the rmgpu
  core, while RMG-Py keeps them pruned in the edge. This is a **screening-
  criterion difference** (fixed-snapshot max(fwd,rev), promote-only, no
  demotion) - NOT a job-07/pdep artifact. Per the user's direction, record this
  as a known rmgpu behavior (more aggressive screening). Whether to align the
  screening criterion is a SEPARATE future step; this step only documents it
  accurately. Do not claim job-07 fixes it.

### Fix 7 - the fast-path must parse the real coverage numbers
`gates/gate_06.py:228-238`. In the skipped-run path, parse `summary.md` for the
estimation + coverage numbers (the file already contains
`library_hits / ml_hits / coverage_errors` and `thermo_errors / kinetics_errors /
species_dropped / reactions_dropped`) and put the REAL values in
`estimation_counts` / `coverage_gaps`. The superminimal numbers are: library 2,
ml 491, coverage_errors 550 (thermo 402, kinetics 148; species_dropped 402,
reactions_dropped 550). Do not hardcode `{}`.

## The honest gate (final pass/fail contract)

After the fixes, `python gates/gate_06.py` must:
- Compute all 6 checks (subgate, superminimal run+parity, c3h4 run+parity,
  output tree, physical validity, provenance).
- Hard-fail (exit 1) if ANY of: subgate != PASS; superminimal run did not
  complete; **c3h4 status != PASS** (seed not loaded / crashed / non-physical);
  output tree missing/invalid; **physical_validity != PASS** (either example);
  provenance not real.
- Record the superminimal + c3h4 core/edge counts, the parity set diffs vs
  `gates/baselines/<example>/summary.json`, and the (fixed) divergence-cause as
  a FINDING. A divergent core with a documented cause is fine (PASS).
- Write `reports/gate_06_results.json` (commit it - jobs 03-05 all committed
  theirs) and print the console summary.

**Honest-outcome note:** it is acceptable (and likely) that the superminimal
final profile is non-physical (O-chain blow-up) even after the c3h4 fixes. If so,
the gate will be RED on `physical_validity` for superminimal. That is a REAL,
honest RED - a valid finding, not a stack failure. In that case:
- the report MUST state exactly: superminimal profile non-physical (which
  species, the max value, at the final iteration), root-caused to the
  promote-only fixed-snapshot screen (Fix 6), and note that aligning the screen
  is a follow-up step (do NOT open it here).
- the c3h4 half MUST be GREEN (seed loaded, run completes, c3h4 profile
  physically valid) - c3h4 is the example whose seed mechanism this step is
  repairing, and it should be clean.
- Set the step row + job-06 status accordingly (c3h4 fixed + honest; superminimal
  profile divergence recorded as a follow-up finding) and leave NEXT at the
  appropriate next step (job-07-step-01, or a new follow-up fix-step if you
  judge the superminimal screen alignment in-scope for a later step).
If, after the fixes, BOTH profiles are physically valid and both runs complete,
the gate is simply GREEN and you document the (qualitative) parity finding.

## Reference to read (this step's budget)

- `gates/gate_06.py` (the file you are fixing - read it all).
- `rmgpu/db/seed_loader.py`, `rmgpu/main.py` (lines ~92-135 seed wiring, ~215-300
  run loop + summary), `rmgpu/core/loop.py` (lines ~409-465 simulate+screen,
  ~468-535 run loop, ~556-565 prune), `rmgpu/reactor/simulator.py` (RateParam,
  simulate_mole_fractions).
- `examples/superminimal.yaml`, `examples/c3h4.yaml` (the inputs - do NOT edit
  them; they are lossless imports).
- `reports/gate_06_results.json` (the false-GREEN evidence), `reports/job-06-
  step-06-gate.md` (the false report you are correcting),
  `gates/baselines/{superminimal,c3h4}/summary.json` (the RMG-Py references:
  superminimal 13 core species / 19 reactions; c3h4 101 core species / 1659
  reactions).
- RMG-Py screen: `rmgpy/rmg/model.py:1418-1455` (how RMG-Py screens).
Do not re-run RMG-Py to regenerate references - the committed baselines are the
reference. Read ONLY the above plus direct dependencies you hit (note extras in
the report).

## Deliverables

- `rmgpu/db/seed_loader.py`: Fix 1 (name resolution), Fix 2 (no placeholder
  methane; loud missing species), Fix 4 (attach thermo when present), Fix 5
  (multi-band Arrhenius - assemble or record), Fix 6 (termolecular A-unit by
  real unit).
- `rmgpu/main.py`: Fix 3 (seed-mechanism load failure is loud, not swallowed).
- `gates/gate_06.py`: Fix 7 (honest hard checks incl. c3h4 + physical
  validity), Fix 8 (fast-path parses real coverage), and the divergence-cause
  block rewritten to match the real run (Fix 6 root cause).
- `tests/`: a seed-loader test (c3h4 seed name -> non-empty species AND
  reactions; missing species does not create methane; termolecular A converted
  by 1e-12; multi-band not silently LIMIT-1) + a physical-validity check test.
- `reports/job-06-step-07-fix-gate.md` (the honest report).
- `reports/gate_06_results.json` (re-run + committed).

## Checks (must run and pass before claiming done)

Run with `/home/jackson/miniforge3/envs/rmgpu/bin/python`:
```
python -m pytest tests/ -q        # all pass (baseline 595; must not drop)
python gates/gate_06.py           # exits 0 ONLY if the honest hard checks pass;
                                  # c3h4 MUST be PASS (seed loaded, run completes,
                                  # c3h4 profile physically valid). superminimal
                                  # profile non-physical => honest RED, documented.
```
Then verify the specific fixes:
- c3h4 seed loads: run the c3h4 example and confirm the core species count is
  well above 3 (GRI seed = 54) and the seed reaction count is non-zero; confirm
  NO methane placeholder appears in a reaction where the label was missing.
- Termolecular A: confirm a `cm^6/(mol^2*s)` reaction's A is scaled by 1e-12
  (not 1e-6) in the loaded seed mechanism.
- Physical validity: confirm the gate reads the final profile row and FAILs if
  any mole fraction is outside [0,1] (a synthetic out-of-range case must FAIL).
- Coverage: confirm the gate's JSON carries the REAL estimation counts (not {}).

## Pitfalls

- Do NOT edit `examples/superminimal.yaml` / `examples/c3h4.yaml` to change the
  seed name or tolerances - they are lossless imports (job-03). Fix the loader
  and the gate, not the inputs.
- Do NOT loosen the physical-validity check or the tolerance to force a pass -
  a non-physical profile is a real finding. Per the user's note, you are not
  required to make the core set identical to RMG-Py, but a blow-up to 250x is
  never "qualitatively similar."
- Do NOT "fix" the O-chain over-generation to match RMG-Py exactly - document it
  as the screening-criterion difference (Fix 6) and leave alignment as a
  follow-up.
- The c3h4 run is LONG (full GRI seed, 1350 K). If it does not reach steady
  state in a reasonable time, the gate still needs: seed loaded + run completes
  (or reaches max_iterations with counts recorded) + c3h4 profile physically
  valid. You may bound it (e.g. a max_iterations cap in the gate or a timeout) as
  long as the seed is demonstrably loaded and the profile is valid - record the
  iteration count + core/edge counts reached.
- Do NOT re-run RMG-Py. The committed baselines are the reference.
- ONE session at a time; no subagents; use the env python
  `/home/jackson/miniforge3/envs/rmgpu/bin/python` only. Never kill processes
  you did not start.

## Done protocol (exact)

1. Commit the code: `git commit -am "job-06/step-07: <summary>"` (several commits
   are fine; NO push). Include the re-run `reports/gate_06_results.json`.
2. STATUS.md: set the 06/07 step row to done (or RED+documented if superminimal
   profile is non-physical), set the 06/06 row + job-06 row to reflect the
   re-opened gate (c3h4 fixed + honest; superminimal finding recorded), append a
   session-log entry, and update the top-level NEXT pointer (to job-07-step-01
   if fully resolved, or to a new follow-up fix-step for the superminimal screen
   if you open one).
3. Write `reports/job-06-step-07-fix-gate.md` with: what was fixed (files + 1
   line each), the checks run (commands + REAL results), the parity findings
   (superminimal + c3h4 set diffs + causes), the coverage numbers, the
   superminimal-profile physical-validity outcome (GREEN or RED+cause), reference
   reads beyond the list, deviations, and what the next step should know first.
4. STOP. Do not start the next step. Do not spawn subagents.
