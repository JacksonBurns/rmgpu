# job-06: Core/edge mechanism loop + torchdae reactor (first integration)

Read ORIENTATION.md (retain #1) and PLAN.md sections 1, 4, 8, 10 (Phase 2), 12.3
(output tree). Prereqs: jobs 01-05. This is the FIRST job where rmgpu runs an actual
mechanism generation - the moment of truth for the stack.

## Goal

The `rmgpu run` command actually works end to end (gas phase, constant T or T/P):
load YAML -> build model from seed species -> estimate properties (libraries/ML) ->
enumerate candidate reactions (recipes) -> simulate in torchdae -> screen by
conversion -> grow/prune -> iterate to steady state -> write the output tree
(PLAN.md 12.3), including the canonical `mechanism/core.yaml`.

## Reference (read the actual RMG-Py code - this is the core of the product)

  RMG-Py/rmgpy/rmg/model.py (114k)  - CoreEdgeReactionModel: enlarge, prune,
    thermo_filter_species, add_reactions_to_model, generate_reactions,
    edge->core promotion, the whole bookkeeping.
  RMG-Py/rmgpy/rmg/main.py (149k)   - the job driver (run_rmg, the iteration loop,
    simulation scheduling).
  RMG-Py/rmgpy/rmg/react.py         - reaction generation driver.
  RMG-Py/rmgpy/solver/simple.pyx (49k), base.pyx (72k) - the reactor ODEs:
    mole balance dN/dt = nu * r, termination conditions (conversion, time,
    criticality), observables, the initial-concentration setup. Port the MATH;
    the solver backend becomes torchdae.
  RMG-Py/rmgpy/tools/decay.py       - the screening/decay logic if separate.
  Termination: RMG-Py/rmgpy/solver/termination.py.

## Deliverables

1. `rmgpu/reactor/reactors.py` - reactor definitions (dataclasses):
   SimpleReactor (isothermal batch), ConstantVReactor, ConstantTPReactor (with
   the energy balance ODE - port from simple.pyx/base.pyx), + stub types
   (Liquid/MBSampled -> job 11, Surface -> job 12). Termination criteria
   (conversion, time) ported exactly (they drive the screen).
2. `rmgpu/reactor/torch.py` - torchdae backend (SOLE solver):
   - build the ODE/DAE right-hand side as a torch function: state = moles (or
     mole fractions + total), y' = nu @ rates(y, T, P), rates from the rate
     registry (job 04), transport NOT needed for gas (ideal).
   - integrate with torchdae (BDF2 or TR-BDF2; record the choice) on cuda if
     available else cpu (single GPU task rule).
   - events/termination: use torchdae events for conversion/time termination;
     observables sampled on a time grid.
   - expose a clean API: simulate(reaction_mechanism, reactor, T, P, t_end,
     termination) -> Profiles (times, species amounts/molefractions,
     termination info). SI units. Mole fractions by default in the output (the
     2.3.0 "moles vs fractions" footgun is fixed, PLAN.md 12.3).
   - VALIDATION SUB-GATE (do this before wiring into the loop): integrate a
     stiff reference ODE (e.g. a 3-reaction Lindemann falloff toy system, or Van
     der Pol mu=10 as a stiff non-chemistry control) and check against a high-
     accuracy reference (RK45 tiny step). Record max abs diff in the report. This
     is PLAN.md 13-risk-3: prove torchdae integrates a real stiff system to
     reference tolerance before trusting it with mechanism growth.
3. `rmgpu/core/model.py` - CoreEdgeReactionModel (port from rmg/model.py):
   - core/edge sets, rates, reactions, species; `enlarge()` (generate reactions
     via job-05 families, estimate kinetics/thermo via job-04 resolvers,
     simulate, screen), `prune()` (rate/thermo-based removal per RMG's rules),
   - thermo_filter (remove species by thermo per RMG's filterReactions/filter
     logic), edge->core promotion by conversion threshold,
   - the model holds the rate registry objects (job 04) - the reaction object
     unifies species + rate model + family/template provenance.
4. `rmgpu/main.py` - the job driver (from rmg/main.py): load resolved YAML (job 03),
   build Databases (job 02), estimators (job 04), families (job 05), reactors
   (1), model (3); run the iteration loop (max iterations / steady-state
   criterion per RMG's), call pdep hook (job 07 - stub: reactions marked
   pressure-dependent get HPL-only kinetics until job 07 lands - this keeps
   this job honest and gas-phase HPL; document that), and WRITE THE OUTPUT TREE
   exactly per PLAN.md 12.3: run.yaml (resolved input), provenance.yaml (git hash
   of rmgpu + rmgdb version + checkpoint hashes + torch/rdkit/cantera versions),
   summary.md (counts, iterations, coverage/ML-vs-library split, warnings),
   rmgpu.log, events.jsonl (per-iteration/per-species events), mechanism/core.yaml
   + edge.yaml (the canonical artifact schema - define it in rmgpu/schemas/
   mechanism.py this job: species with formula/smiles/adjlist/source/thermo,
   reactions with reactants/products/family/source/degeneracy/kinetics model+
   params+method(ml|library)), species/<label>.json (+ .svg via rdMolDraw2D),
   reactions/reactions.json, profiles/<reactor>/time_series.csv + metadata.yaml.
   (pdep/uncertainty/iterations dirs: created by later jobs; make the output
   writer extensible.)
5. `rmgpu/schemas/mechanism.py` - the canonical artifact schema (pydantic):
   loadable back into a model (bidirectional seed, PLAN.md 12.3) and exportable
   to Chemkin (job 08) / Cantera (job 08, via Cantera API or ck2yaml of a
   generated chem.inp).
6. `rmgpu/io/chemkin.py` - WRITER for chemkin.inp + chem_annotated.inp +
   species_dictionary.txt from the canonical artifact (port from RMG-Py's
   chemkin.pyx WRITE path; reading not needed yet).

## Gate (job 06) -> gates/gate_06.py, report reports/job-06.md

  1. torchdae sub-gate (above): stiff ODE integration accuracy recorded.
  2. `rmgpu run` on the imported `superminimal` (examples/rmg/superminimal) with
     HPL kinetics (pdep off/stub): completes N iterations to steady state (or
     max_iter). Record iteration count, core/edge species+reaction counts.
  3. Parity: compare final core species set + core reaction set (canonical SMILES
     keys) against RMG-Py run on the same input (run RMG-Py with pdep OFF - use a
     modified copy of its input; save its output to gates/baselines/superminimal/).
     Report: |core_rmgpu - core_rmg|, |core_rmgpu + edge - (core+edge)_rmg| as
     sets (lists of species/reactions on each side that differ).
     TARGET: core identical (or documented divergence with cause); edge within a
     small fraction. Any systematic divergence (a family always missing/extra) is
     a BLOCKER to fix in this job if tractable, else recorded precisely.
  4. Same for `c3h4` (examples/rmg/c3h4) - a real mechanism (do NOT use the
     full GRI-scale example yet; c3h4 is the right size).
  5. Output tree: verify every file in PLAN.md 12.3 exists and is valid
     (YAML parses, CSV has the right columns, core.yaml loads back via the schema
     and re-simulates the final iteration's profiles within tolerance).
  6. Provenance: provenance.yaml contains real hashes/versions (not placeholders).

## When done

STATUS.md (job 06 row + log entry with the core/edge parity numbers) + commit
"job-06: core/edge loop + torchdae reactor; first mechanism generation" + STOP.

## Pitfalls

- The core/edge bookkeeping in rmg/model.py is intricate (edge reactions,
  reaction rates for screening, the "interrupt" tolerance). Port the logic
  exactly; the parity gate depends on it.
- torchdae API: check the installed version's actual API (0.1.1): the
  integrator call signature, event syntax, adjoint (job 09). If the API differs
  from the plan's description, adapt and document.
- Screening uses reaction RATES at the current conditions - make sure the rate
  registry's k(T,P) is fast (numpy/torch, no per-iteration Python loops over
  reactions if avoidable - but do NOT micro-optimize prematurely; correctness
  first, profile later).
- Do not wire pressure dependence in this job (HPL only); the pdep hook is a
  documented stub so job 07 plugs in without touching the loop.
- Memory: mechanisms grow; keep the profile history bounded (RMG keeps the last
  few; do the same).
