# RMG-GPU (RMGPU) -- Feasibility & Feature-Parity Plan (FINAL)

This is the final plan. v2 incorporated four notes: (1) this is a research bet /
proof-of-concept, so we go
all-in on ML property estimators + GPU DAE solvers and do NOT spend LOC on fallbacks;
(2) rate-rule kinetics are also replaced by Chemprop (reaction prediction verified in
source); (3) Arkane and QM are dropped from the runtime entirely - and QM is now out of scope
in every capacity, since model improvement belongs outside the package (see §7/§8a); (4) catalysis/solvation stay plugins, but this doc now contains a detailed
enough design that we *could* build them.

Everything is grounded in the clean `RMG-Py` (v4.0.0-5-gd08392ed), `RMG-database`,
`rmgdb` copies here, plus a live check of the replacement packages and a source read of
chemprop 2.3.1. LOC figures are measured, not estimated.

---

## 1. Verdict (feasibility)

**Feasible as a proof of concept, and the all-in-on-ML framing makes it *smaller*, not
bigger, than a conservative rewrite.**

The core insight that makes this work: RMG's runtime is a pipeline, and the two
"estimation" legs of that pipeline (thermo and high-pressure-limit kinetics) are exactly
the two things a foundation model predicts. If Chemprop/CheMeleon covers those, the two
biggest *algorithmic* subsystems in RMG -- group additivity/HBI and the rate-rule
Bayesian-modeling tree -- are deleted, not ported. What remains to retain is the
orchestration (the core/edge loop), the reaction *generation* machinery (recipes,
resonance, atom types), and the *pressure-dependence engine* (the master equation), which
is a simulation, not a prediction, so ML does not touch it.

Concretely, the pipeline becomes:

```
                [ ML: Chemprop/CheMeleon ]
   species ---->  thermo (Hf298, S298, Cp(T))
                [ ML: Chemprop reaction model ]
   reaction --->  high-pressure-limit (HPL) kinetics (A, n, Ea or k(T) grid)
                                   |
            [ master equation: GPU numerics, NOT ML ]
              pressure dependence: k(T,P) falloff from HPL + stat mech (DoS, collision)
              stat mech inputs: ML Hf298 (E0) + group frequencies (statmech DB)
              TS E0: derived from the HPL rate when conformer absent (RMG behavior)
                                   |
                [ reactor: torchdae DAE, GPU/CPU ]
              integrate dX/dt, screen, prune  ---->  core/edge mechanism loop
```

- ML replaces **estimation**. It does not replace the **master equation** (pressure
  dependence is a simulation that consumes the ML-predicted HPL rate as an input anchor)
  or the **reactor integrator** (a stiff DAE). Those two stay as GPU numerics.
- Because this is a PoC, there are **no fallback estimators**: no group-additivity path,
  no rate-rule path, no scipy reactor path. torch already gives us the CPU fallback for
  the solver at zero extra LOC. If the ML models don't cover something, that is a
  research *finding to report*, not a reason to bolt on a legacy code path.

The genuinely-retained core is therefore the smallest it can be: **the core/edge
orchestration + reaction-recipe generation + resonance/atom-type machinery + the
master equation - and no QM at all (model improvement happens outside the package;
the checkpoint interface is the only seam, §8a.3).** Realistic retained-and-modernized
code: **~25-40k LOC** (down from my v1 ~35-50k because the rate-rule tree and the
GA/HBI estimators are deleted rather than retained).

---

## 2. What RMG actually is (measured, unchanged from v1)

```
RMG-Py  rmgpy  95,969 LOC   (py + pyx + pxd)
        arkane  14,486 LOC   (separate package -- dropped from the runtime, see §7/§8a)
RMG-database  2,418,049 LOC of *data*
rmgdb         SQLite standard + built .db files (thermo.db 11 MB, kinetics.db 14 MB, ...)
```

`rmgpy` breakdown (measured): molecule/ 21.7k, data/ 18.9k, rmg/ 11k, tools/ 8.5k,
kinetics/ 6.2k, solver/ 5.2k, pdep/ 4.5k, statmech/ 4.3k, thermo/ 2.5k, qm/ 2.4k.

Database: 135 kinetics families (~60 are `Surface_*` catalysis), 74 kinetics libraries,
77 thermo libraries, plus depository, solvation, statmech, transport, surface DBs. `rmgdb`
already packages all of it into SQLAlchemy schemas / SQLite, with a typed table for every
rate model RMG uses (Arrhenius, ArrheniusEP, ArrheniusBM, Marcus, Troe, Lindemann,
ThirdBody, Chebyshev, PDepArrhenius, SoluteTSDiff, Efficiencies).

---

## 3. What gets DELETED (replaced by ML) -- the big win

| Subsystem (where) | Was | Becomes |
|---|---|---|
| Group additivity + HBI radical thermo (`data/thermo.py`, ~1k of the 3.1k) | Benson GA over the group DB + hydrogen-bond-increment radical correction | **Chemprop/CheMeleon** thermo (Hf298, S298, Cp(T)) |
| Rate-rule + Bayesian-modeling tree (`data/kinetics/rules.py` + ~1.5k of `family.py`) | Template->rate averaging, information-gain tree fit over training reactions | **Chemprop reaction model** (predicts HPL rate / Arrhenius params per reaction) |
| The "try each resonance isomer with each method" fallback chains in thermo/kinetics lookup | Priority-sorted multi-method estimation | **Single ML estimator**; libraries are the only override |

Note: **libraries and the depository are NOT deleted** -- curated experimental/computed
data still takes priority over ML. What's deleted is the *algorithmic backends* (GA math,
rate-rule tree) and the fallback orchestration around them. The retrieval logic
("is this species/reaction in a library? if not, estimate") stays, but the "estimate"
branch is one call into the ML model, not a cascade.

### 3a. The Chemprop-based estimators are REPLACED, not reused

RMG-Py already ships a Chemprop-based thermo estimator: `rmgpy/ml/estimator.py`
(`MLEstimator`, a Chemprop species-model wrapper, currently disabled on py3.11 --
upstream issue #2559). It is the historical *seam* for ML in RMG, and in rmgpu it is
**replaced, not re-enabled**:

- **New code, old checkpoint layout.** rmgpu's estimators (`rmgpu/ml/thermo_estimator.py`,
  `rmgpu/ml/kinetics_estimator.py`) are re-implemented from scratch. The reference
  implementation to follow is `chemprop_example/predicting.ipynb` (in the workspace):
  `models.MPNN.load_from_checkpoint(checkpoint_path)`, the model's featurizer,
  `data.MoleculeDatapoint` / `MoleculeDataset` / `data.build_dataloader`, and
  `pl.Trainer(accelerator=...).predict(model, loader)`. RMG's `ml/estimator.py` is
  read *only* to learn (a) the checkpoint layout it consumes (separate Hf298 model
  and S298+Cp model), (b) its uncertainty cutoffs, and (c) how the `ml_estimator:`
  input DSL names them. Nothing is imported, copied, or wrapped from it -- it is
  reference material, like the rest of RMG-Py's source.
- **Existing checkpoints are consumed, not their infrastructure.** The current
  checkpoints (CheMeleon thermo models; any reaction-model checkpoints) keep
  working, but they are loaded through the *new* estimators' own load path (which
  mirrors chemprop_example's), not through RMG's `MLEstimator`. If a checkpoint
  does not load via that pattern, the new load path takes an explicit featurizer
  argument and the mismatch is recorded as a finding (job 04, step 01).
- **Kinetics estimators have no RMG counterpart.** RMG's wrapper was species-only;
  the reaction (HPL kinetics) estimator is net-new in rmgpu, built on chemprop's
  native reaction mode (6, verified in the 2.3.1 source).
- **Checkpoint interface unchanged.** A better/future checkpoint is a config change
  (the `ml_estimator:` block: name + hash), never a code change -- the same seam as
  8a.3. The replacement changes *how checkpoints are loaded*, not *who supplies them*.

---

## 4. What gets RETAINED (and modernized to numpy/torch, no Cython)

These are RMG's actual IP and the reasons the rewrite isn't just "call Chemprop."
(QM is NOT in this list and is out of scope entirely -- see §8a: it has no runtime role
in the ML-first design, and model improvement, including any QM-based data generation,
belongs outside the package.)

1. **Core/edge mechanism loop** -- `rmg/rmg/model.py` (`CoreEdgeReactionModel`:
   `enlarge / prune / thermo_filter_species / add_*`). Rate-based growth, edge->core
   promotion, thermodynamic filtering. The heart of the product.
2. **Reaction *recipe* DSL** -- `data/kinetics/family.py` (`ReactionRecipe`,
   `apply_recipe`, `_generate_product_structures`, ~1.5-2k LOC). A custom atom-labeled
   bond-ops language that enumerates products from reaction templates and tracks
   degeneracy. **Not SMARTS reactions** and not replaceable by RDKit `ReactionFromSmarts`.
   Retain; re-express on the new Molecule wrapper.
3. **Resonance-structure generation** -- `molecule/resonance.py`. RDKit's resonance set
   != RMG's, and thermo/kinetics can depend on which isomer is canonical. Retain.
4. **Atom-type DB + assignment** -- `molecule/atomtype.py` (RMG-specific atom types over
   an RDKit graph). Retain, thin.
5. **Master equation / pressure dependence** -- `pdep/network.py` (1149) +
   `collision.pyx`, `configuration.pyx` + collision models (`cse`, `msc`, `rs`, `sls`).
   Discretized master equation: density of states, microcanonical rates, collision-energy
   handling, four lumping methods. **This is a simulation, retained as GPU numerics**
   (torch for the big sparse matrix ops/eigenproblems, torchdae for the time-dependent ME
   integration). Top GPU target after the reactor.
6. **Stat mech** -- `statmech/` (conformer, hindered/free rotation, 1D/2D torsion rotors,
   vibration, translation). Standard physics; the 1D/2D rotor PDE solves move to
   scipy/torch. Feeds the master equation (DoS, collision, TS energies) with data from
   the statmech DB (group frequencies), ML Hf298, and the HPL rate (TS E0) -- no QM
   required at runtime (see §8a.2).
7. **Chemkin I/O** -- `chemkin.pyx` (2446) or Cantera `ck2yaml`. Table-stakes interop.

**Everything else in `rmgpy` is delegation-eligible** (molecule graph ops -> RDKit,
database I/O -> rmgdb, kinetics expressions -> numpy/torch, reactor -> torchdae,
transport -> ChEDL, units -> pint).

---

## 5. What gets DELEGATED (verified available)

| RMG subsystem (LOC) | Replace with | Status |
|---|---|---|
| Molecule graph ops, isomorphism, substructure, degeneracy, canonicalization, SMARTS (~15k of molecule/) | **RDKit** | Verified: canonicalization, `GetSubstructMatch` (degeneracy), SMARTS parse work on installed RDKit 2024.09.4. Thin `Molecule` wrapper over `RWMol`. |
| Graph isomorphism / VF2 (`vf2.pyx`) | **RDKit** (subgraph) or **graph-tool** (C++ VF2) | Delete `vf2.pyx`. |
| All database I/O (`data/base.py` 1445 + every entry load/save) | **`rmgdb`** (SQLAlchemy/SQLite + YAML) | Your wrapper; the single biggest I/O win; typed tables for every rate model. |
| Thermo property prediction | **Chemprop / CheMeleon** | REPLACED in rmgpu (not re-enabled): the new estimators are a fresh re-implementation of the checkpoint+inference infrastructure per chemprop_example (see 3a); RMG's `ml/estimator.py` is reference-only. |
| Kinetics (HPL) property prediction | **Chemprop** (reaction models) | Verified in source, see §6. |
| Kinetics rate *expressions* (arrhenius 2046, falloff, chebyshev, tunneling, `kinetics/model.pyx`) | **numpy/torch** (thin model registry, no Cython) | Pure math; ML predicts the parameters, the registry wraps them into k(T,P) the reactor/ME consume. |
| Reactor DAE solver (`solver/` 5.2k, Cython + PyDAS) | **torchdae** (sole backend; CPU+GPU via device) | Verified: PyPI 0.1.1, BDF/TR-BDF2/Radau-IIA, index reduction, adjoint, vmap. No scipy path (torch has the CPU fallback). |
| Sensitivity / uncertainty (`tools/uncertainty.py` 1792) | **torch autodiff / adjoint** (torchdae adjoint) | Your point (1). |
| Transport (Brokaw/Wilke/LJ/diffusion; `data/transport.py` 677) | **`chemicals`/`fluids`/`thermo`** (ChEDL) | Verified on PyPI (1.5.2/1.3.1/0.6.1). |
| Units / dimensional analysis (`quantity.py` 886, cimported everywhere) | **`pint`** | Removes the Cython `Quantity` woven through every Cython module. |
| NASA-7/9 + Cp eval (`thermo/wilhoit.pyx`, `nasa.pyx`) | **Cantera** NASA + **`thermo`**; keep a thin Wilhoit in numpy | Wilhoit (2005) is RMG-specific and small; keep it, delegate the rest. |
| Chemkin I/O + Cantera export | **Cantera** `ck2yaml` + thin writer | RMG already imports `ck2yaml`. |

---

## 6. Chemprop reaction prediction (verified in the 2.3.1 source)

You asked me to convince myself kinetics can be ML-predicted. It can. From the actual
chemprop 2.3.1 package (downloaded and read):

- `chemprop/types.py`: `Rxn = tuple[Mol, Mol]` -- a reaction is a (reactant, product)
  mol pair.
- `chemprop/featurizers/molgraph/reaction.py`: `class RxnMode(EnumMapping)` with six
  featurization modes -- `REAC_PROD`, `REAC_PROD_BALANCE`, `REAC_DIFF`,
  `REAC_DIFF_BALANCE`, `PROD_DIFF`, `PROD_DIFF_BALANCE` -- each defining how the
  reactant/product graphs are combined into a single featurizable graph (e.g. concat
  reactant+product, or reactant + the reactant-to-product *difference*, with optional
  stoichiometric balancing).
- `chemprop/data/datapoints.py`: `ReactionDatapoint` and `LazyReactionDatapoint` (the
  lazy one carries `rct_smiles` / `pdt_smiles`).
- `chemprop/cli/train.py`: `--reaction-columns`, `--rxn-mode`, and `rxn_mode` threaded
  through featurization, training, and prediction.
- The models module docstring: "Generate predictions for the input molecules/reactions."

So the kinetics path is: **feed the reaction (as atom-mapped reactant + product, in the
`RxnMode` that best matches the rate rules' feature basis) to a Chemprop reaction model
-> get the predicted HPL rate (or Arrhenius parameters A, n, Ea) -> wrap it in the
rate-expression registry -> feed the master equation for pressure dependence.** This is a
drop-in replacement for `get_kinetics(..., estimator='rate rules')`. The architecture is
the same one RMG half-built in `ml/estimator.py` (which was species-only; the reaction
generalization is exactly what chemprop now natively supports) - but per 3a, rmgpu
does NOT reuse that wrapper: the reaction estimator is written fresh on the
chemprop_example inference pattern, which chemprop's reaction mode plugs into directly
(featurizer + `ReactionDatapoint` + the same load/predict path).

**Architectural note (important for the PoC):** ML gives the *high-pressure-limit* rate.
*Pressure dependence* (falloff, the master equation) is still a simulation that consumes
that HPL anchor. So "all-in on ML for kinetics" does **not** remove the master equation;
it removes the *rate-rule tree* that produced the HPL anchor. Same split as thermo: ML
replaces estimation, the ME replaces nothing (it's the pressure engine).

---

## 7. Arkane & QM: drop both from the runtime

You want the Arkane/RMG difference knocked down. Here's the finding that makes it clean:
**RMG already has its own QM driver in `rmgpy/qm`** (gaussian + mopac parsers,
`QMCalculator`, symmetry detection, statmech->thermo). RMG's *runtime* only imports four
thin things from Arkane:

- `arkane.common.symbol_by_number` (a one-liner helper)
- `arkane.ess.factory.ess_factory` (used by `statmech/ndTorsions.py` for 2D torsion scans)
- `arkane.pdep.PressureDependenceJob` (a job-orchestration convenience in `rmg/input.py`)
- `arkane.output.prettify` (a log-formatting helper in `rmg/main.py`)

So the Arkane dependency surface is tiny, and the *rest* of Arkane (the standalone
thermo-estimation workflow, the ESS batch-submit machinery, the `explorer`/methoxy
exploration, most of `encorr`) is exactly what a mechanism-generation PoC does **not**
need. The deeper question - "do we need QM at all, and how does it relate to the master
equation?" - is answered by tracing the runtime data paths, in §8a.

**Decision (revised after tracing the runtime paths, see §8a.2-8a.3):**
- **We do not mainline QM into the core runtime.** The runtime data path for thermo
  (ML) and pressure dependence (ML thermo + statmech DB + HPL-derived TS energies)
  has no QM dependency. The only Arkane import at runtime (`ess_factory`, used by
  `ndTorsions` for 2D torsion scans) is also dropped, because hindered-rotor barriers
  come from the statmech DB in the ML-first design.
- **QM is entirely out of scope - no runtime, no offline tool.** Model improvement
  (including any QM-based training-data generation) is done entirely outside RMG-GPU,
  by the people who own the models. The seam is the checkpoint interface
  (`ml_estimator:` block: name + hash). If Phase 1 shows a coverage gap, the fix is a
  better checkpoint, not QM plumbing inside the package.
- We do **not** depend on Arkane as a package and do not ship it. Nothing from RMG-Py's
  `qm/` module is carried over. The v2 "slim `rmgpu/qm`" idea, and the later "optional
  offline data-gen tool" idea, are both retired - QM is out of scope, full stop.

This removes a whole second package from the dependency graph and from the "what do I
install" surface, which is the difference you wanted killed.

---

## 8. Target architecture (no fallbacks)

```
rmgpu/
  core/
    model.py            # CoreEdgeReactionModel: enlarge/prune/thermo_filter          [RETAIN #1]
    recipe.py           # reaction recipe DSL + product enumeration + degeneracy     [RETAIN #2]
    atomtype.py         # RMG atom-type DB + assignment (thin, over RDKit)           [RETAIN #4]
    resonance.py        # resonance-structure generation rules                      [RETAIN #3]
  molecule/
    molecule.py         # Molecule = RWMol + atomtype map + labels + resonance list  (RDKit wrapper)
  db/
    loaders.py          # ALL I/O through rmgdb (thermo/kinetics/transport/solvation/statmech)
  ml/
    base.py             # shared Chemprop load/predict path (per chemprop_example)   [3a]
    thermo_estimator.py # Chemprop/CheMeleon species model -> Hf298,S298,Cp(T)         [SOLE estimator; REPLACES rmgpy/ml/estimator.py, 3a]
    kinetics_estimator.py # Chemprop reaction model -> HPL rate / Arrhenius params     [SOLE estimator; net-new]
  kinetics/
    models.py           # thin numpy/torch rate-expression registry (wraps ML output)
  pdep/
    network.py          # master equation: DoS, microcanonical rates, CSE/MSC/RS/SLS   [RETAIN #5, torch]
    collision.py        # collision-energy models (torch)
  statmech/             # conformer/rotation/vibration/torsion; rotors via torch/scipy  [RETAIN #6]
  (no qm/ anywhere)     # QM out of scope entirely; model improvement happens outside RMG-GPU  [§8a.3]
  reactor/
    torch.py            # torchdae backend (GPU/CPU), adjoint sensitivity            [SOLE solver]
    reactors.py         # simple / constant-V / constant-T-P / liquid / mb-sampled
  transport.py          # chemicals/fluids/thermo backed
  io/
    chemkin.py          # read/write (thin) or Cantera ck2yaml
    cantera_yaml.py     # export
  units.py              # pint-backed thin Quantity
  input.py / main.py / output.py   # input DSL, job driver, HTML/plot output
  plugins/              # extension points (see §9) -- surface + solvation live here
```

Dependencies (verified): `torch` (CUDA), `numpy`, `rdkit`, `pint`, `chemicals`, `fluids`,
`thermo`, `cantera`, `torchdae`, `chemprop` + CheMeleon checkpoints, `rmgdb` (yours).
Optional: `polars` for bulk DB reads. **No Cython, no numba** (numba only if a specific
master-equation kernel is proven hot; torch covers it).

---

## 8a. Pressure dependence (master equation) and QM: what they actually are in RMG today

This section exists to settle two things the plan above elided: (i) what the master
equation / pressure-dependence machinery actually does inside RMG, and (ii) what QM is
really used for -- because the short answer to "do we need QM at runtime?" is **no**, and
the plan now says so explicitly.

### 8a.1 What the master equation computes

Unimolecular reactions (A -> products or A -> isomers) are in principle *barrier
crossing* events whose rate depends on how much internal energy the molecule has. At
low pressure the molecule keeps its energy between collisions (LVA limit), so
k(T,P) is pressure-dependent: below the fall-off, k grows with P; above it, k saturates
at the high-pressure limit k_inf(T). The **discretized master equation** (Bauer +
Rice-Ramsperger-Kassel theory, as implemented in RMG) is the standard way to get k(T,P)
across the whole fall-off:

- Discretize the internal energy into "grains" (bins).
- Build a rate matrix: diagonal = unimolecular transitions out of a state (RRKM
  microcanonical rates, which need the **transition-state density of states** and the
  **energy gap** to the TS), off-diagonal = collisional energy transfer between grains
  (collision models: CSE, MSC, RS, SLS).
- Integrate the population vector in time (this is a DAE/ODE in the energy populations)
  until it reaches the quasi-stationary state that corresponds to a steady reactive flux.
- The flux out of the reactive channels is the phenomenological k(T,P).

RMG does this **per pressure-dependent family** (H_ABstraction, R_Addition_*,
R_Reactions, Intra_H_Abstraction, Intra_R_Addition, Cyclic_Elimination,
Isomerization, ...), building **networks**: a connected set of isomers + reactant/
product channels that share intermediates. For each (T,P) point in a grid, the network
is solved to produce k(T,P) for every reaction in it, which is then fit to a
Chebyshev or PDepArrhenius polynomial and stored as the reaction's `falloff` kinetics.
Those fit coefficients are what the reactor actually integrates.

**Inputs the master equation needs, per species/isomer/TS:**
- E0 (ground-state energy, = ZPE-corrected electronic energy) -- for E0 of each isomer
  and the TS.
- Vibrational frequencies + rotor barriers -- to build the density of states rho(E)
  for isomers and the TS density of states rho_TS(E).
- Collision model parameters (LJ sigma/epsilon, bath gas).

### 8a.2 Where those inputs come from in RMG today (the key part)

This is where the QM question resolves. I traced the actual code paths:

- **E0 of isomers**: from the species' **thermo** object (Hf298), which comes from
  libraries, group additivity, or (in RMG-GPU) ML. Not from QM at runtime.
- **Vibrational frequencies**: from the **statmech database** (characteristic group
  frequencies, e.g. "ringCH -> 2750-3150 cm^-1 x 2"), fit to the species' heat capacity.
  `rmgpy/data/statmech.py::get_statmech_data` is the runtime path; it needs NO QM --
  it uses the statmech DB group frequencies plus the thermo model's Cp(T).
- **TS density of states**: RMG does NOT compute TS frequencies via QM in the runtime
  path. In `arkane/pdep.py` (which RMG's `pressure_dependence()` DSL builds on) the
  transition-state E0 is **approximated from the HPL rate** when no conformer is
  attached: `transition_state.conformer.E0 = sum(reactant E0) - RT*ln(k_inf*V/h)`
  (lines 278-282). This is the standard "derive the barrier from the high-pressure-limit
  rate" trick -- it closes the loop: the ML-predicted HPL rate (our Chemprop output)
  supplies the TS energy for the master equation.
- **Collision parameters**: LJ sigma/epsilon from the transport DB (LJ correlations or
  experimental), not QM.

So the runtime data path for the master equation in RMG-GPU is:

```
   isomer:  E0  <- ML thermo (Hf298)         [no QM]
            freqs <- statmech DB group freqs  [no QM]
            rotors<- statmech DB / fit        [no QM]
   TS:      E0  <- derived from HPL rate (ML kinetics)  [no QM]
            freqs <- same statmech DB path       [no QM]
   collision: LJ from transport DB             [no QM]
```

**QM is not on the runtime path.** The only place QM touches the core in RMG-Py is
`rmgpy/data/kinetics/family.py:1250-1252`, inside the **rate-rule training** step
(depository processing), where it can optionally compute QM thermo for training
species -- and even that is gated behind the `quantum_mechanics(...)` DSL flag, which
is off by default. The `run_jobs` call writes QM *input* files and reads back computed
**thermo** (Hf298/S298/Cp), not frequencies for the master equation. The `quantum_mechanics`
DSL function's only other consumer is Arkane's independent QM workflow.

### 8a.3 Decision: QM is entirely out of scope for RMG-GPU

The honest accounting of where QM could matter:

- **Runtime estimation** (thermo, HPL kinetics): ML (Chemprop/CheMeleon). QM plays no
  role.
- **Runtime pressure dependence**: master equation fed by ML thermo + statmech DB +
  HPL-derived TS energies. QM plays no role.
- **Training-data generation** for the ML models: this is model development, not
  mechanism generation. It belongs with the people who own the models (Chemprop/
  CheMeleon development), done entirely outside RMG-GPU. RMG-GPU consumes model
  checkpoints; it does not build them.

**Decision: RMG-GPU contains no QM code, of any kind.** No parsers, no QM output
reading, no offline data-generation tool, no `quantum_mechanics` input block. Users who
want to improve the ML models do so entirely outside the package - the checkpoint
interface (name + hash, in `ml_estimator:` of the input) is the entire seam between
model improvement and mechanism generation.

This is a deliberate scope boundary, not an oversight. RMG's core is reactor
simulation + reaction/species enumeration + property estimation. QM was never that in
modern RMG either (off-by-default, training-only); keeping any trace of it would add
parsers, dependencies, and a second data path for zero runtime benefit. If a specific
coverage gap shows up in the Phase 1 ML-accuracy gate, the response is to improve the
ML models (outside this package) and re-point the checkpoint, not to add QM plumbing.

The 2D-torsion-scan need (the one Arkane import in RMG's runtime) also disappears:
hindered-rotor barriers come from the statmech DB (group values) or the rotor-fit path,
so no Arkane import survives.

## 9. Plugin design: catalysis & solvation (we don't build these first, but here's how)

The core loop in `model.py` does five things a plugin must be able to inject into:
(1) **generate** candidate reactions, (2) **assign kinetics**, (3) **assign thermo**,
(4) **represent** species, (5) **simulate** in a reactor. A clean plugin protocol exposes
exactly these as extension points on the core, so a plugin is a bag of providers + hooks,
and the core never knows which plugins are loaded.

### 9.1 The plugin interface (what the core exposes)

```python
class RmgpuPlugin:
    name: str
    # --- registration (called once at model init) ---
    def register_species(self, registry)        -> ...   # add Species subtypes
    def register_families(self, loader)         -> ...   # add reaction families/recipes
    def register_thermo(self, db)               -> ...   # add/override a thermo source
    def register_kinetics(self, db)             -> ...   # add a kinetics source / rate model
    def register_reactor(self, factory)         -> ...   # add reactor types
    # --- per-object hooks (called inside the core loop) ---
    def on_new_species(self, species, model)    -> list  # return reactions/species to enqueue
    def on_new_reaction(self, rxn, model)       -> ...   # adjust kinetics/thermo assignment
    def after_prune(self, model)                -> ...   # cleanup / bookkeeping
    # --- data ---
    def load_database(self, rmgdb_handle)       -> ...   # pull this plugin's tables
```

The core calls these in a fixed order and merges the results into the core/edge sets. A
plugin that only touches thermo (solvation) implements two methods; a full phase
(catalysis) implements all of them. That asymmetry is the whole point: the protocol is
uniform, the implementations are sized to the phase.

### 9.2 Catalysis plugin (the big one)

Catalysis is a *second phase*, so it needs the most of the protocol. What it must
provide, mapped to RMG's existing surface code (which is the reference for what a full
implementation looks like):

- **Species: `SurfaceSpecies` = adsorbate + site.** A surface species is an adsorbate
  bound to a surface site (the site is a small site-graph, e.g. a metal atom + neighbor
  atoms). RMG's reference: `rmgpy/species.py` surface handling + the `Surface_*`
  families' reactants. The plugin's `register_species` adds this type so the core loop
  can hold it in core/edge like any species.
- **Reaction families: the ~60 `Surface_*` families.** These are *reaction recipes*
  (same DSL as the core) that operate on surface species: adsorption
  (`Surface_Adsorption_Single/Double/vdW/Dissociative`), dissociation, migration,
  abstraction (Eley-Rideal and Langmuir-Hinshelwood), beta-scission, proton/electron
  transfer, carbonate formation/decomposition, etc. The plugin's `register_families`
  loads these recipes from the surface DB (rmgdb already has a surface section). They run
  through the *same* `recipe.py` engine as the core families -- the plugin only supplies
  the recipes + site templates, not a new generator.
- **Thermo: adsorption energies + coverage dependence.** `register_thermo` provides a
  source that, for a surface species, returns the adsorption energy (from the adsorbate
  library) plus, if requested, **coverage-dependent** corrections. RMG's reference:
  `ThermoDatabase.correct_binding_energy` (linear-scaling correction of adsorption
  energies against a reference metal) and `set_binding_energies`. This is a thermo
  *provider*, not a new thermo engine.
- **Kinetics: coverage-dependent surface rate models.** `register_kinetics` provides the
  surface rate expressions (RMG's reference: `rmgpy/kinetics/surface.pyx`, 1291 LOC --
  coverage-dependent pre-exponential and the site-balance terms). These wrap into the
  same `kinetics/models.py` registry the core uses; the surface models just carry extra
  coverage-theta terms.
- **Reactor: the surface reactor.** `register_reactor` adds a reactor that, in addition
  to gas/liquid mole fractions, tracks **site coverage** (fractions of vacant vs occupied
  sites) and solves the site-balance (a small algebraic constraint -- this is where the
  DAE index > 1 appears, and exactly what torchdae's index-reduction is for). RMG's
  reference: `rmgpy/solver/surface.pyx` (1110) + `SurfaceReactor`. In the PoC this is a
  torchdae system with the site balance as an algebraic (algebraic) constraint.
- **Database:** `load_database` pulls the adsorbate library + surface families from
  rmgdb's surface section (which mirrors RMG-database's `input/surface/libraries` and the
  `Surface_*` family definitions).

**Effort shape:** most of the surface *families* are data (recipes) that flow through the
existing recipe engine, so the plugin's novel code is really (a) the `SurfaceSpecies`
type, (b) the site-graph representation, (c) coverage-dependent thermo/kinetics
providers, and (d) the coverage-tracking reactor. That's a bounded, self-contained
package that imports the core's recipe/kinetics/pdep modules rather than forking them.
This is why it's a plugin and not core: it reuses the recipe engine, the ML estimators
(adsorption-energy models are just another Chemprop species model), and the reactor
interface.

### 9.3 Solvation plugin (small)

Solvation is a *correction layer*, not a new phase, so it implements only the thermo +
kinetics providers:

- **Species:** none new (solved species are the same gas-phase species, just in solution).
- **Thermo provider:** `register_thermo` adds a source that returns the **solvated**
  thermo = gas-phase (from ML) + a solvation free-energy correction. RMG's reference:
  `rmgpy/data/solvation.py` (2455 LOC) -- implicit-solvation (SMD-like) corrections
  parameterized in the solvation DB (rmgdb has `solvation.db`). The provider is a
  (gas_thermo, solvation_db) -> solvated_thermo function the core's thermo resolver
  consults when the run is in solution.
- **Kinetics provider:** `register_kinetics` adds the solvation correction to rates
  (reaction free-energy-of-solvation shifts the barrier). Same DB.
- **Reactor:** the `LiquidReactor` / `MBSampledReactor` reactor types (RMG's
  `solver/liquid.pyx`, `solver/mbSampled.pyx`) are registered so the core can simulate
  liquid-phase runs. These are torchdae systems too.
- **Database:** `load_database` pulls the solvation groups/libraries from rmgdb.

**Effort shape:** small -- it's two provider functions + a couple of reactor types + a
DB load, all reusing core machinery. This is the reference for how *light* a plugin can
be, and it's a good first plugin to validate the protocol with before attempting
catalysis.

### 9.4 Why this is credible "we could"

- The core's recipe engine, ML estimators, rate registry, pdep, and reactor interface are
  all **shared** -- a plugin is data (recipes/DB) + a few providers + hooks, not a fork.
- The protocol is the same five verbs the core already performs, so adding a plugin does
  not change the core's invariants (core/edge bookkeeping stays in `model.py`).
- Both catalysis and solvation have a direct line of sight to existing RMG reference code
  (the `Surface_*` families, `kinetics/surface.pyx`, `solver/surface.pyx`,
  `data/solvation.py`, `solver/liquid.pyx`), so "enough detail to build it" is anchored
  in known-good implementations, not speculation.
- Ordering: solvation first (validates the protocol cheaply), catalysis second (the big
  one), both strictly after the gas/liquid core reaches parity.

---

## 10. Feature-parity roadmap (PoC-optimized, no fallback phases)

Parity is defined the same way RMG defines it: re-run the existing
`test/regression/<name>/input.py` set + `scripts/checkModels.py` and diff core/edge
models and profiles. The only change from v1 is that there are no "add the fallback"
phases -- ML is in from Phase 1, and torch is the only reactor.

**Execution note:** the phases below are realized as a sequence of JOBS (job-00 to
job-12, one per the prompts/ job briefs), each decomposed into STEPS (prompts/steps/,
one fresh human-started session per step - see the repo README for the loop). A phase may
span several jobs (e.g. Phase 3 = jobs 07-08); a job may span several sessions (its
steps). The gates below are the job gates.

### Phase 0 -- Foundations + validation harness
- `rmgpu` package; conda env (py>=3.11, torch+cuda, rdkit, cantera, chemicals/fluids/
  thermo, pint, torchdae, chemprop, rmgdb, CheMeleon checkpoints).
- Thin layers first: `units.py` (pint), `molecule/` (RDKit wrapper + atomtype DB +
  resonance + adjlist/SMILES round-trip), `db/loaders.py` (everything via rmgdb).
- **Gate:** round-trip every molecule in `superminimal`/`c3h4`/`methylformate`; assert
  RMG-adjlist == rdkit-canonical; assert rmgdb loads the same entries RMG-Py loads
  (entry counts + a hash of the kinetics/thermo tables).

### Phase 1 -- Static property evaluation (ML is the whole thing)
- `ml/base.py` (the shared Chemprop load/predict path, per
  chemprop_example/predicting.ipynb - 3a) + `ml/thermo_estimator.py`
  (CheMeleon/Chemprop; REPLACES rmgpy/ml/estimator.py, not a re-enable) +
  `ml/kinetics_estimator.py` (Chemprop reaction model) + `kinetics/models.py`
  registry + `transport/`.
- **Gate:** for a fixed set of species/reactions (including intermediates and TS-adjacent
  species), assert predicted Hf298/S298/Cp(T) and HPL k(T) against RMG-Py's values
  (libraries where present; ML where not). **This is the PoC's thesis test** -- if the
  ML estimators don't match RMG's library+GA+rate-rules accuracy here, the PoC's premise
  fails early, which is exactly when you want to find out. Report the error distribution,
  don't hide it. Note: the master equation's TS energy is derived from the ML HPL rate
  (§8a.2), so this gate also validates the pressure-dependence data path end-to-end.

### Phase 2 -- Mechanism generation (core loop), gas-phase
- `core/` (model loop, recipe, atomtype, resonance) + `reactor/torch.py` (torchdae).
- **Gate:** run `superminimal` and `c3h4` to a fixed iteration count; diff core/edge
  species + reaction sets vs RMG-Py baselines. Target: identical core, edge within
  documented tolerance. Highest-value milestone.

### Phase 3 -- Pressure dependence (master equation, GPU)
- `pdep/` (DoS, microcanonical rates, collision, CSE/MSC/RS/SLS in torch) + `statmech/`
  (rotors via torch). Data path: ML Hf298 (E0) + statmech DB group frequencies (DoS) +
  HPL-derived TS E0 + LJ collision params -- **no QM** (§8a.2).
- **Gate:** run an `R_Addition_MultipleBond`-heavy example (e.g. `propane_branching`);
  assert k(T,P) falloff curves and pdep-network products match RMG-Py. The numerically
  hardest phase; port CSE (the default) first, diff, then add the others.

### Phase 4 -- Full core parity
- Liquid (`LiquidReactor`), isotope, uncertainty (torchdae adjoint), observables
  regression, diff/merge models, flux diagrams.
- **Gate:** full `test/regression` suite green via `checkModels.py`; all 38 examples
  reproduce. **This is feature parity for the gas/liquid core.**

### Phase 5 (post-parity, out of scope for the PoC) -- plugins
- Solvation plugin (first, cheap, validates the protocol), then the catalysis plugin
  (the big one). See §9.

### Effort intuition
- Phases 0-2: delegation + careful porting; most LOC reduction happens here; low risk
  *except* Phase 1's ML-accuracy gate (the thesis).
- Phase 3 (master equation): the most expensive *correctness* item.
- Phase 4: the remaining calendar time for core parity, mostly plumbing.
- Phase 5: the long tail, strictly deferred.

---

## 11. Retain / delete / external map (v2)

| Subsystem | Decision | Why |
|---|---|---|
| Core/edge loop (`model.py`) | **RETAIN** | The product. |
| Reaction recipes (`family.py`) | **RETAIN** | Custom atom-label DSL; not SMARTS. |
| Rate rules + BM tree | **DELETE** | Chemprop reaction model replaces it. |
| Group additivity + HBI | **DELETE** | Chemprop/CheMeleon replaces it. |
| Thermo/kinetics *retrieval* (library/depository search + "else ML") | **RETAIN (thin)** | Libraries still override ML; only the estimation branch collapses to one ML call. |
| Master equation / pdep | **RETAIN + torch** | Pressure dependence is a simulation, not a prediction. |
| Stat mech | **RETAIN + torch** | Feeds the ME (DoS, collision, rotors); data from statmech DB, no QM (§8a.2). |
| Resonance rules | **RETAIN** | RDKit's set differs. |
| Atom-type DB + assignment | **RETAIN (thin)** | RMG-specific types over RDKit. |
| Molecule graph ops | **EXTERNAL (RDKit)** | isomorphism/substructure/canonical/SMARTS verified. |
| VF2 | **EXTERNAL (RDKit/graph-tool)** | Delete `vf2.pyx`. |
| Database I/O | **EXTERNAL (rmgdb)** | Biggest I/O win. |
| Kinetics rate expressions | **REWRITE (numpy/torch)** | Pure math; ML predicts params, registry wraps them. |
| Reactor solver | **EXTERNAL (torchdae, sole)** | torch gives the CPU fallback free. |
| Sensitivity/uncertainty | **EXTERNAL (torch adjoint)** | Your point (1). |
| Transport | **EXTERNAL (ChEDL)** | Verified on PyPI. |
| Units | **EXTERNAL (pint)** | Kill the cimported `Quantity`. |
| Chemkin I/O | **EXTERNAL (Cantera ck2yaml) + thin** | Interop. |
| Thermo models (NASA/Wilhoit) | **MIXED** | NASA via Cantera/`thermo`; thin numpy Wilhoit. |
| Thermo/kinetics *estimation* | **EXTERNAL (Chemprop/CheMeleon)** | Sole estimators; REPLACED (3a): fresh re-implementation per chemprop_example; RMG's `ml/estimator.py` (#2559) is reference-only. |
| QM / Arkane | **OUT OF SCOPE (nothing carried over)** | No runtime role (§8a). Model improvement (incl. any QM data-gen) is done outside the package; the checkpoint interface is the seam. No Arkane dependency. |
| Surface/catalysis (~60 fam) | **PLUGIN (defer, §9.2)** | Reuses recipe engine + ML + reactor interface. |
| Solvation | **PLUGIN (defer, §9.3)** | Lightest plugin; first to validate the protocol. |
| MBSampledReactor (liquid) | **REACTOR (Phase 4)** | torchdae. |

---

## 12. Input & output (I/O) redesign

RMG's I/O is as much a usability problem as its code is a maintainability problem, so
this gets redesigned as part of the rewrite, not bolted on. The current state (measured
from the tree + docs):

- **Input**: a Python file that is *executed* to configure the run. 41 top-level DSL
  functions (`database`, `species`, `simpleReactor`, `constant_V_ideal_gas_reactor`,
  `liquid_reactor`, `mb_sampled_reactor`, `surface_reactor`, `simulator`, `solvation`,
  `model`, `quantum_mechanics`, `ml_estimator`, `pressure_dependence`, `uncertainty`,
  `options`, `forbidden`, `smiles`, `adjacency_list`, `react`, `restart_from_seed`, ...).
  Executing arbitrary Python means: no schema, no validation, config and code are fused,
  and a typo is a runtime exception deep in the pipeline, not a readable error.
- **Output**: `output.html` (one big opaque blob), `/chemkin` (`chem.inp`,
  `chem_annotated.inp`, `chem_NNNN.inp` per-iteration snapshots,
  `species_dictionary.txt`), **three** Cantera-YAML routes (`/cantera_from_ck` + beta
  `/cantera1` + beta `/cantera2`), `/rms`, `/pdep`, `/solver` (CSV of mole *amounts*),
  `/species` (png), `/plot`, `RMG.log`, `covariance.csv`. No single canonical
  machine-readable mechanism artifact; provenance is implicit.

There is an open upstream PR in the right direction (#2844, "Add yaml input file
support" -- a `YAMLInputReader` + extension auto-detection). We adopt its *direction*
(YAML, config/code separation, validation, tooling) but not its specific class; we go
further with a typed schema, provenance, a single canonical mechanism artifact, and a
one-shot legacy importer.

### 12.1 Design principles (both input and output)

1. **Declarative data, never code.** The run is a data document, schema-validated before
   anything executes. No `exec`/`eval` of user files.
2. **One canonical mechanism artifact.** A prior run's mechanism is *both* an output and
   a seed input (`restart_from_seed`, `seedMechanisms`). So the mechanism format is
   bidirectional and the single source of truth; every legacy format (Chemkin, Cantera
   YAML, RMS) becomes an *on-demand export* derived from it, not a primary. This kills
   the 3-Cantera-route mess.
3. **Provenance by default.** Every run writes the exact resolved input + the code/DB/
   model versions that produced it, so any run is bit-for-bit reproducible.
4. **Typed, with units first-class.** Quantities carry units (pint-backed); a mistyped
   quantity is a validation error at load, not a NaN halfway through.
5. **Human + machine.** A generated one-page report + per-object JSON, alongside a
   structured JSONL log. No more "the HTML blob is the only nice view."
6. **Deterministic tree.** Same input -> same file tree (sorted, stable), so CI can diff
   outputs and the regression harness can do its job.

### 12.2 New input format

A single YAML document (`input.yaml`), validated against a **pydantic** schema (this is
the core UX win: real type/field errors with locations, not runtime crashes). The 41
camelCase DSL functions collapse into ~12 top-level keys, each an ordered block:

```yaml
# input.yaml
rmgpu: 1.0                     # schema version (forward-compat gate)
extends: ./base.yaml           # optional: layered config (standard conditions, per-fuel overrides)

database:
  thermo_libraries:   [primaryThermoLibrary]
  reaction_libraries: []
  seed_mechanisms:    [my_mechanism/core.yaml]   # path to a prior run's mechanism artifact
  kinetics_families:  default
  kinetics_depositories: [training]
  kinetics_estimator: ml          # 'ml' (Chemprop) -- the PoC default; 'library' also allowed
  transport_libraries: auto

species:
  - { label: ethane, reactive: true, structure: "CC" }                # SMILES
  - { label: CH3CHO, reactive: true, structure: "CC=O" }
  - { label: H,      reactive: true, structure: "[H]" }
  # adjacency list still accepted: structure: { adjlist: "1 C u0 {2 S, ...}" }
  # per-species overrides allowed: thermo: {library: X}, constraints: [...]

forbidden:
  - { structure: "C=[C]=[C]", reason: "cumulenes unsupported" }

reactors:                        # polymorphic on `type`
  - type: simple
    temperature: { value: 1350, unit: K }
    pressure:    { value: 1.0,  unit: bar }
    initial_mole_fractions: { ethane: 1.0 }
    termination: { conversion: { species: ethane, value: 0.9 }, time: { value: 1e6, unit: s } }
  # type: const_V | const_TP | liquid | mb_sampled | surface (each its own typed model)

simulator: { atol: 1e-16, rtol: 1e-8 }

model:
  tolerance_keep_in_edge: 0.0
  tolerance_move_to_core: 0.1
  tolerance_interrupt_simulation: 0.1
  maximum_edge_species: 100000
  filter_reactions: true

pressure_dependence: { method: cse }          # cse|masc|rs|sls; network grain controls, etc.
ml_estimator:      { thermo: chemeleon_thermo_v1, kinetics: chemeleon_rxn_v1 }   # checkpoint refs
# (no quantum_mechanics block: QM is out of scope entirely, §8a - model improvement
#  happens outside the package)
solvation:        { solvent: acetonitrile, model: smd }                          # optional
uncertainty:      { enabled: true, species: [CO, CO2] }                          # optional
options:          { save_profiles: true, save_plots: false, save_edge: true, units: si }
```

Design specifics:
- **Quantity syntax**: `{value, unit}` maps, or the compact `"1350 K"` string that pint
  parses. The legacy `(1350, 'K')` tuple is importable (see 12.3) but the new canonical
  form is the map.
- **No Python objects**: `structure` is a string (SMILES / InChI) or an explicit
  `{adjlist: "..."}`; no `SMILES(...)` constructors, no dicts-of-objects.
- **Separation of concerns**: a run can reference external files by path
  (`seed_mechanisms: [path]`, `thermo: {file: ...}`), so a big seed mechanism and a small
  run config stay separate.
- **Layering via `extends`**: resolve `extends` chains into one flat document at load
  (left-most wins); enables a shared "conditions.yaml" + per-fuel overrides.
- **Tooling**: the pydantic models export a **JSON Schema** (`rmgpu schema`) for editor
  autocomplete/LSP; `rmgpu validate input.yaml` runs with zero side effects and reports
  every problem at once.
- **Backward-compatible importer** (one-shot, `rmgpu import old.py --to new.yaml`):
  parses the legacy Python DSL with **`ast`, not `exec`** (safe, no code runs) and maps
  the 41 functions onto the schema. This preserves all 38 examples and every user's
  existing inputs, and auto-detection by file extension (`.py` -> importer, `.yaml` ->
  native) gives the same convenience as PR #2844.

**CLI** (replaces the `rmg.py`/`Arkane.py` exec entrypoints):
```
rmgpu run input.yaml          # run a job
rmgpu validate input.yaml     # schema-check only, no side effects
rmgpu import old.py           # legacy Python DSL -> new YAML
rmgpu export core.yaml --to chemkin|cantera|rms   # canonical -> legacy formats
rmgpu diff runA/core.yaml runB/core.yaml
rmgpu inspect runDir/         # print the provenance + summary of a finished run
```

### 12.3 New output structure

A single, versioned, hierarchical artifact tree per run (root = run name or timestamp):

```
run/
  run.yaml                 # the EXACT resolved input that produced this run (reproducible)
  provenance.yaml          # rmgpu git, rmgdb version+hash, ML checkpoint hashes,
                           # solver/units settings, timestamp, python/torch/rdkit versions
  summary.md               # one-page human report (core/edge counts, iterations,
                           #    warnings, ML-vs-library coverage, provenance)  [+ summary.html optional]
  rmgpu.log                # human-readable log
  events.jsonl             # structured log (per-iteration, per-species, per-reaction events)

  mechanism/
    core.yaml              # <-- THE canonical machine-readable mechanism (species+reactions)
    edge.yaml              # edge model (same format)
    chemkin.inp            # legacy interop export (== old chem.inp)
    chemkin_annotated.inp  # annotated export
    species_dictionary.txt # adjacency lists (kept for tooling compat)
    cantera/chem.yaml      # Cantera YAML -- ONE route (drops from_ck/cantera1/cantera2)
    rms.yaml               # RMS export (optional)

  species/
    <label>.svg            # 2D structure (svg primary, png on request)
    <label>.json           # {formula, smiles, adjlist, source, symmetry,
                           #   thermo: {Hf298, S298, Cp(T) grid, method: ml|library, uncertainty}}
  reactions/
    reactions.json         # [{reactants, products, family, source, degeneracy,
                           #   k: {model, A, n, Ea, Tgrid, Pgrid, method: ml|library, uncertainty}}]

  profiles/
    <reactor>/
      time_series.csv      # well-formed: time[s], <species1_molefrac>, <species2>, ...
      metadata.yaml        # conditions, termination, solver params, units
  sensitivity/             # torchdae adjoint results (when enabled)
  pdep/
    <network>.yaml         # pdep network def + k(T,P) grid (self-contained; feeds nothing else)
  uncertainty/
    covariance.csv
    parameter_uncertainties.json
  iterations/              # OPTIONAL, off by default; per-iteration mechanism snapshots + manifest
```

Key changes vs. today, and why:
- **`mechanism/core.yaml` is the single canonical artifact** (stable, documented,
  versioned schema). Chemkin / Cantera / RMS are *exports* of it, generated on demand
  (`rmgpu export`) or at run-end if requested -- no more three competing Cantera routes,
  no more "which chem.inp is the real one."
- **A mechanism is bidirectional**: `run/mechanism/core.yaml` is directly usable as
  `seed_mechanisms` in a new `input.yaml`. Clean restart-from-seed, no adjacency-list
  round-trip fragility.
- **Profiles are schema'd** (mole fraction by default, with an explicit `units` field --
  fixing the 2.3.0 "these are moles now, divide by the sum" footgun in the docs).
- **Provenance is written every time** (`run.yaml` + `provenance.yaml`); reproducibility
  is the default, not a feature you remember to turn on.
- **The HTML blob is demoted**: `summary.md`/`summary.html` is the primary human entry
  point; per-species/per-reaction JSON replaces "dig through the big HTML" for programmatic
  access; the old `output.html` (if anyone still wants it) becomes a thin generated view
  over the same JSON.
- **Structured logging**: `events.jsonl` (machine) + `rmgpu.log` (human), replacing the
  single unstructured `RMG.log`.
- **No scattered `chem_NNNN.inp` snapshots** in the top dir; per-iteration history goes
  under `iterations/` (opt-in, with a manifest) so the default tree is small and stable.

### 12.4 Schemas are shared & versioned

The **input schema** (pydantic) and the **mechanism artifact schema** are defined
together in one place (`rmgpu/schemas/`), versioned via the `rmgpu: 1.0` key, and are
the de-facto API of the package. Because rmgdb is the data standard, the *database* side
of the schema is a thin pointer (library/family names resolve against the rmgdb build
named in `provenance.yaml`), so the input stays small while the DB stays out of the
document. This is what lets us keep the 2.4M-line database external and still have a
tiny, reviewable, diffable input.

### 12.5 Where this lands in the roadmap

- **Phase 0** grows to include: define the pydantic input + mechanism schemas, the
  `rmgpu` CLI, the legacy `ast`-based importer, and the provenance writer. Gate: the 38
  example `.py` inputs import to `.yaml` losslessly, and `validate` passes on all of them.
- **Phase 2** (core loop) is where the new output tree first matters: the regression
  harness (`checkModels.py` equivalent) diffs `mechanism/core.yaml` across RMG-Py and
  RMG-GPU instead of parsing Chemkin -- which is itself a parity win, because the
  comparison no longer depends on the fragile chem.inp/adjacency-list round-trip.
- The output redesign is **not** a separate phase; it is the I/O layer that Phases 0-4
  all build on, sized so the Phase-0 gate (lossless import + validation) is reached early.

---

## 13. Risks & open questions (v2, honest)

1. **The ML-accuracy gate is the whole PoC.** With no fallback, the success of the PoC
   is *defined* by whether Chemprop/CheMeleon thermo + Chemprop reaction kinetics match
   RMG's library+GA+rate-rules accuracy on mechanism generation. Phase 1 is built to
   expose this early. There is no safety net by design; the deliverable is the measured
   error distribution + the generated mechanisms, not a fallback that hides a gap.
2. **Coverage of intermediates / transition states by the ML models.** ML thermo models
   are usually trained on stable species, but the master equation needs Hf298/S298/Cp
   (for E0) for *intermediates* (and the TS energy, derivable from the HPL rate). If the
   ML models don't cover intermediates, the pressure-dependence leg starves -- and there
   is no runtime QM fallback by design. This is the sharpest technical risk. Mitigation
   (research, not code, and entirely OUTSIDE this package): train/validate the CheMeleon
   models on intermediates using whatever data-generation tooling the model team uses;
   RMG-GPU's only interface to all of that is the checkpoint name + hash. If the gap
   proves unfixable, that is a PoC result to report, not a feature to bolt on.
3. **torchdae maturity.** Right shape (GPU, differentiable, vmap, index reduction,
   adjoint) but young (0.1.1, 4 stars). It's the *only* reactor backend now (no scipy
   path). If it regresses on a stiff chemical system, there's no second backend -- so
   Phase 2 should include an explicit "does torchdae integrate a real combustion ODE to
   the same tolerance as a reference" check before we trust it with mechanism growth.
4. **Master-equation correctness (Phase 3).** CSE/MSC/RS/SLS are subtle; a naive rewrite
   silently diverges. Mitigation: port CSE first, diff k(T,P) falloff vs RMG-Py on a
   dedicated pdep set, then add the others one at a time.
5. **Resonance-isomer ordering.** RDKit can't reproduce RMG's resonance set; thermo/
   kinetics can depend on which isomer is canonical. Keep `resonance.py` as-is and only
   delegate graph *operations*, not *generation*.
6. **rmgdb round-trip completeness.** Your README validates the *kinetics* DB round-trip
   explicitly; confirm thermo/transport/solvation/statmech are equally clean before Phase
   0 makes rmgdb the single I/O path.
7. **Data is the moat.** 2.4M LOC of database is untouched and is the real value. Phase 0's
   hash gate enforces bit-compatibility with it; rmgdb is what makes that cheap.
8. **Legacy input import completeness.** The `ast`-based importer must cover all 41
   DSL functions including the long tail (surface reactor blocks, solvation, `react`
   tuples, staged reactors, liquid mass-transfer coefficients). Risk: the legacy DSL has
   implicit Python semantics (dict ordering, `auto` sentinels, nested `SMILES()`) that a
   schema must either model or reject explicitly. Mitigation: the Phase-0 gate is
   lossless round-trip on all 38 shipped examples *plus* a diff of every imported YAML
   against the original `input.py`'s parsed `ast` (same values, no drops); any DSL
   feature the importer can't express is documented and fails loudly, never silently
   defaulted.
9. **Canonical artifact churn.** Making `core.yaml` the bidirectional seed format means
   it must stay stable and versioned. Mitigation: the `rmgpu: 1.0` version key + schema
   in `rmgpu/schemas/` is the API contract; breaking changes are major-version bumps with
   an auto-migrator, and the regression harness diffing core.yaml (§12.5) doubles as the
   stability test.

---

## 14. Anti-goals (v2)

- No Cython, no numba-by-default (numba only if an ME kernel is proven hot).
- No fallback estimators (no GA, no rate-rules), no second reactor backend. The ML models
  and torchdae are the single source of truth; gaps are reported, not papered over.
- No re-implementing graph isomorphism, transport correlations, units, or the DAE solver.
- No reusing RMG-Py's Chemprop wrapper (`rmgpy/ml/estimator.py`): it is reference-only;
  rmgpu's estimators are a fresh re-implementation per chemprop_example (3a).
- No QM in any form (no parsers, no offline data-gen tool): model improvement is done
  outside the package; the checkpoint interface is the only seam (§7/§8a.3).
- No shipping Arkane as a package; nothing mainlined from it.
- No catalysis/solvation in the core package -- they are plugins (§9), built only after
  core parity.
- No "modernizing" the 2.4M-line database format (rmgdb already did that).

---

### Appendix A -- Verification notes (what I actually ran)
- LOC: `find ... | xargs wc -l` per module (figures in §2).
- **chemprop 2.3.1 source** (downloaded, read): `types.py` `Rxn = tuple[Mol, Mol]`;
  `featurizers/molgraph/reaction.py` `RxnMode` enum (REAC_PROD, REAC_PROD_BALANCE,
  REAC_DIFF, REAC_DIFF_BALANCE, PROD_DIFF, PROD_DIFF_BALANCE); `data/datapoints.py`
  `ReactionDatapoint` / `LazyReactionDatapoint` (`rct_smiles`/`pdt_smiles`);
  `cli/train.py` `--reaction-columns`, `--rxn-mode`, `rxn_mode` threading. Confirms
  reaction (not just species) prediction.
- torchdae: PyPI 0.1.1 (py>=3.8) + README (BDF1/2, SDIRK TR-BDF2, Radau-IIA, Pantelides
  index reduction, CPM/Baumgarte, vectorized events, adjoint, vmap).
- scipy: `solve_ivp` BDF & Radau on stiff Van der Pol (mu=10) -- both converge (reference
  only now; not a backend).
- RDKit 2024.09.4: canonicalization, `GetSubstructMatch` (degeneracy), SMARTS parse --
  verified. Reaction engine ships as `rdChemReactions.so` but Python bindings aren't
  auto-registered in this conda build; not load-bearing (recipes are a custom DSL).
- chemicals 1.5.2 / fluids 1.3.1 / thermo 0.6.1 / cantera 3.2.0 / sundials 7.8.0 /
  chemprop 2.3.1 / pint 0.25.3 / polars 1.43.2 / torch 2.11.0+cu130 (CUDA available).
- RMG `ml/estimator.py` read in full: a chemprop *species* wrapper, disabled for py3.11
  (issue #2559) -- the seam to generalize to reactions.
- RMG `qm/` module read (gaussian/mopac parsers, QMCalculator, symmetry, statmech->thermo);
  grep confirms RMG's only Arkane imports are `symbol_by_number`, `ess_factory`
  (ndTorsions 2D scans), `PressureDependenceJob`, `output.prettify` -- basis for §7.
- rmgdb `kinetics/schema.py` + `thermo/schema.py` read in full; SQLite files present
  (thermo.db 11 MB, kinetics.db 14 MB, + transport/solvation/statmech).
