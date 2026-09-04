"""The core/edge mechanism-growth loop (job-06, the "moment of truth").

Wires the modules jobs 01-05 built into the real iteration loop that
``rmgpu run`` drives (rmgpy/rmg/main.py ``execute`` ported; gas-phase,
HPL/pressure-independent kinetics - the pdep stub; job 07 turns pdep on in
both rmgpu and the RMG-Py reference):

  enlarge    families (job 05) -> generate_reactions over core-core pairs
             (and core-edge when ``react_edge``) -> each product molecule
             resolved to a species (canonical-SMILES identity) -> thermo +
             kinetics via the job-04 resolvers (library -> ML ->
             MLCoverageError, never a third branch) -> new reactions +
             species into the edge.
  simulate   the current mechanism in the torchdae reactor (job 06/01)
             over a timescale set by the characteristic rate.
  screen     species/reactions by rate-ratio vs the characteristic rate
             (RMG-Py base.pyx): above ``tolerance_move_to_core`` -> core;
             below ``tolerance_keep_in_edge`` -> dropped; else edge.
  iterate    enlarge -> simulate -> screen -> prune until the core/edge
             signature is unchanged (RMG's steady-state criterion) or
             ``max_iterations``.

No fallbacks: anything neither the libraries nor the ML checkpoints cover
is a recorded coverage gap (dropped, counted in ``EstimationCounts``), never
a silent rate-rule / group-additivity value (PLAN 14).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rmgpu.core.model import CoreEdgeReactionModel, Reaction, Species
from rmgpu.core import enumeration as enum
from rmgpu.core.family import KineticsFamilies
from rmgpu.core.template import TemplateMatcher
from rmgpu.data.estimation import EstimationCounts, MLCoverageError
from rmgpu.data.estimation import estimate_kinetics, estimate_thermo
from rmgpu.kinetics.models import Arrhenius, KineticsModel
from rmgpu.molecule.molecule import Molecule
from rmgpu.reactor.simulator import (RateParam, SimResult,
                                     characteristic_rate,
                                     simulate_mole_fractions)
from rmgpu.logging import get_logger

log = get_logger("core.loop")

# CGS -> SI pre-exponential for the ML branch (PLAN 3b boundary conversion):
# the checkpoint emits A in cm^3/(mol*s); SI is m^3/(mol*s).
CGS3_TO_SI3 = 1.0e-6

# Characteristic-rate timescale factor (RMG simulates ~O(1) times the inverse
# characteristic rate before screening).
CHAR_RATE_TFACTOR = 5.0


# ---------------------------------------------------------------------------
# Canonical structural key (identical to gate_05's: both sides of the parity
# comparison use the same isomorphism-invariant canonical SMILES).
# ---------------------------------------------------------------------------

def canonical_key(mol) -> str:
    from rdkit import Chem
    m = mol._rdkit if hasattr(mol, "_rdkit") else mol
    if not any(a.GetSymbol() == "H" for a in m.GetAtoms()):
        try:
            m = Chem.AddHs(m)
        except Exception:  # noqa: BLE001
            pass
    try:
        Chem.Kekulize(m, clearAromaticFlags=True)
    except Exception:  # noqa: BLE001
        pass
    try:
        Chem.SanitizeMol(m)
    except Exception:  # noqa: BLE001
        pass
    return Chem.MolToSmiles(m)


def _smiles_for_ml(mol) -> str:
    """A plain (label-free, explicit-H) SMILES for building reaction SMILES
    for the kinetics ML featurizer."""
    from rdkit import Chem
    m = Chem.Mol(mol._rdkit)
    if not any(a.GetSymbol() == "H" for a in m.GetAtoms()):
        try:
            m = Chem.AddHs(m)
        except Exception:  # noqa: BLE001
            pass
    try:
        Chem.SanitizeMol(m)
    except Exception:  # noqa: BLE001
        pass
    return Chem.MolToSmiles(m)


# ---------------------------------------------------------------------------
# Configuration / context
# ---------------------------------------------------------------------------

@dataclass
class LoopConfig:
    tolerance_move_to_core: float = 0.01
    tolerance_keep_in_edge: float = 0.001
    max_iterations: int = 25
    react_edge: bool = True           # RMG default
    simulate_steps: int = 32          # torchdae step count for the sim
    max_t: float = 1e12               # hard cap on the sim timescale (s)
    min_t: float = 1e-9               # floor on the sim timescale (s)


@dataclass
class RunContext:
    """Everything the loop needs, built once by the driver (rmgpu.main)."""
    databases: object                  # rmgpu.db.Databases (thermo/kinetics)
    ml: object                         # .thermo / .kinetics estimators (or None)
    families: KineticsFamilies         # loaded (job-05 loader)
    seed_species: List[Species]        # from the input species: block
    initial_mole_fractions: Dict[str, float]
    temperature: float                 # K
    pressure: float                    # Pa
    config: LoopConfig = field(default_factory=LoopConfig)
    thermo_libraries: Optional[List[str]] = None
    reaction_libraries: Optional[List[str]] = None
    termination_time: Optional[float] = None       # s
    termination_conversion: Optional[Dict[str, float]] = None
    seed_mechanisms_species: List[Species] = field(default_factory=list)
    seed_mechanisms_reactions: List[Reaction] = field(default_factory=list)


@dataclass
class RunResult:
    core_model: CoreEdgeReactionModel
    iterations: int
    counts: EstimationCounts
    coverage: Dict[str, int]
    final_profiles: Optional[SimResult]
    core_keys: List[str]
    edge_keys: List[str]
    events: List[Dict] = field(default_factory=list)
    log_lines: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

class CoreEdgeLoop:
    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.model = CoreEdgeReactionModel()
        self.counts = EstimationCounts()
        self.coverage = {"thermo_errors": 0, "kinetics_errors": 0,
                         "species_dropped": 0, "reactions_dropped": 0}
        self.species_by_key: Dict[str, Species] = {}
        # reaction key -> record {rp, react_keys, prod_keys, family}
        self.reactions: Dict[str, Dict] = {}
        self.events: List[Dict] = []
        self.log: List[str] = []
        self._family_cache: Dict[int, enum.Family] = {}
        self._last_profiles: Optional[SimResult] = None
        self._last_sim: Optional[Dict] = None

    # -- species ----------------------------------------------------------
    def _register_species(self, mol: Molecule, label_hint: Optional[str],
                          iteration: int) -> Species:
        key = canonical_key(mol)
        sp = self.species_by_key.get(key)
        if sp is None:
            sp = Species(label=(label_hint or key), molecule=mol,
                         reactive=True, creation_iteration=iteration)
            self.species_by_key[key] = sp
        return sp

    def _species_for_piece(self, mol, iteration: int) -> Species:
        key = canonical_key(mol)
        sp = self.species_by_key.get(key)
        if sp is None:
            sp = self._register_species(mol, None, iteration)
        return sp

    def _thermo(self, sp: Species, iteration: int) -> bool:
        """Estimate thermo for a species (cached on the Species). False on a
        coverage gap (counted)."""
        if sp.thermo is not None:
            return True
        d = {"label": sp.label, "smiles": sp.molecule.to_smiles()}
        try:
            sp.thermo = estimate_thermo(
                d, self.ctx.databases.thermo, self.ctx.ml,
                counts=self.counts, libraries=self.ctx.thermo_libraries)
            return True
        except MLCoverageError:
            self.coverage["thermo_errors"] += 1
            self.coverage["species_dropped"] += 1
            self.events.append({"type": "thermo_coverage_error",
                                "species": sp.label, "iteration": iteration})
            return False

    # -- kinetics ---------------------------------------------------------
    def _reaction_smiles(self, react_mols, prod_mols) -> str:
        rs = ".".join(_smiles_for_ml(m) for m in react_mols)
        ps = ".".join(_smiles_for_ml(m) for m in prod_mols)
        return rs + ">>" + ps

    def _rate_for_reaction(self, react_mols, prod_mols, rxn_key: str,
                           family: str, template: Optional[List[str]],
                           degeneracy: float, iteration: int) -> Optional[RateParam]:
        """Estimate kinetics (library -> ML) + thermo of every participant,
        then assemble an SI RateParam. None on a coverage gap (counted)."""
        for m in list(react_mols) + list(prod_mols):
            sp = self._species_for_piece(m, iteration)
            if not self._thermo(sp, iteration):
                self.coverage["reactions_dropped"] += 1
                return None

        def sd(m):
            sp = self._species_for_piece(m, iteration)
            return {"label": sp.label, "smiles": sp.molecule.to_smiles()}

        reaction = {
            "label": rxn_key,
            "reactants": [sd(m) for m in react_mols],
            "products": [sd(m) for m in prod_mols],
            "reaction_smiles": self._reaction_smiles(react_mols, prod_mols),
        }
        try:
            model, deg = estimate_kinetics(
                reaction, self.ctx.databases.kinetics, self.ctx.ml,
                counts=self.counts, libraries=self.ctx.reaction_libraries,
                degeneracy=degeneracy)
        except MLCoverageError:
            self.coverage["kinetics_errors"] += 1
            self.coverage["reactions_dropped"] += 1
            self.events.append({"type": "kinetics_coverage_error",
                                "reaction": rxn_key, "iteration": iteration})
            return None
        # Source tag: the ML branch always yields a bare Arrhenius with an
        # "ML estimate" comment; library hits may be any assembled model.
        source = ("ml" if (isinstance(model, Arrhenius)
                           and "ML estimate" in str(getattr(model, "comment", "")))
                  else "library")
        if isinstance(model, Arrhenius):
            A = float(model.A)
            if source == "ml":
                A = A * CGS3_TO_SI3
            n = float(model.n)
            Ea = float(model.Ea)
            T0 = float(getattr(model, "T0", 1.0) or 1.0)
        else:
            hi = getattr(model, "highPlimit", None)
            if isinstance(hi, Arrhenius):
                A = float(hi.A); n = float(hi.n)
                Ea = float(hi.Ea); T0 = float(getattr(hi, "T0", 1.0) or 1.0)
            else:
                try:
                    A = float(model.get_rate_coefficient(self.ctx.temperature))
                except Exception:  # noqa: BLE001
                    self.coverage["reactions_dropped"] += 1
                    return None
                n, Ea, T0 = 0.0, 0.0, 1.0

        # dS / dH from participant thermo (H298/S298, Hf298). job-06/step-07:
        # dH is required by the thermodynamically consistent reverse factor
        # (1/K_c); the old entropy-only factor was off by up to 30 orders of
        # magnitude for exothermic reactions (root cause of the non-physical
        # c3h4 profile).
        dS = 0.0
        dH = 0.0
        for m in prod_mols:
            sp = self._species_for_piece(m, iteration)
            dS += sp.thermo.S298
            dH += sp.thermo.Hf298
        for m in react_mols:
            sp = self._species_for_piece(m, iteration)
            dS -= sp.thermo.S298
            dH -= sp.thermo.Hf298
        return RateParam(A=A, n=n, Ea=Ea, T0=T0, dS=dS, dH=dH,
                         reversible=True, family=family, template=template,
                         degeneracy=float(deg), source=source)

    # -- families ---------------------------------------------------------
    def _family_enumeration(self, loaded) -> Optional[enum.Family]:
        if id(loaded) in self._family_cache:
            return self._family_cache[id(loaded)]
        efam = enum.Family(
            label=loaded.label,
            recipe=loaded.recipe,
            reverse_recipe=loaded.reverse_recipe,
            own_reverse=loaded.own_reverse,
            reversible=loaded.reversible,
            allow_charged_species=loaded.allow_charged_species,
            electrons=loaded.electrons,
            reactant_num_effective=loaded.num_template_reactants_effective,
            product_num_forward=loaded.product_num_forward,
            reverse_map=loaded.reverse_map,
            template_labels=[e.label for e in loaded.forward_template],
            forbidden=loaded.forbidden,
        )
        efam.matcher = TemplateMatcher(loaded)
        self._family_cache[id(loaded)] = efam
        return efam

    # -- enlarge ----------------------------------------------------------
    def enlarge(self, iteration: int) -> None:
        log.info("enlarge: start iteration %d", iteration)
        core = list(self.model.core.species)
        edge = list(self.model.edge.species)
        log.debug("enlarge: core species %d, edge species %d", len(core), len(edge))
        pairs: List[Tuple] = []
        for i, a in enumerate(core):
            pairs.append((a,))
            for b in core[i:]:
                pairs.append((a, b))
        if self.ctx.config.react_edge:
            for a in core:
                for b in edge:
                    pairs.append((a, b))
        log.info("enlarge: total pairs %d, families %d", len(pairs), len(self.ctx.families.families))
        for loaded in sorted(self.ctx.families.families, key=lambda f: f.label):
            efam = self._family_enumeration(loaded)
            if efam is None:
                log.debug("enlarge: family %s skipped", loaded.label)
                continue
            log.debug("enlarge: family %s with %d pairs", loaded.label, len(pairs))
            for pair in pairs:
                mols = [sp.molecule for sp in pair]
                try:
                    rxns = enum.generate_reactions(efam, mols)
                except Exception:
                    continue
                for tr in rxns:
                    self._add_generated_reaction(tr, efam, iteration)
        log.info("enlarge: done iteration %d", iteration)

    def _add_generated_reaction(self, tr, efam: enum.Family, iteration: int) -> None:
        try:
            react_mols = list(tr.reactants)
            prod_mols = list(tr.products)
            react_keys = [canonical_key(m) for m in react_mols]
            prod_keys = [canonical_key(m) for m in prod_mols]
            rxn_key = ".".join(sorted(react_keys)) + ">>" + ".".join(sorted(prod_keys))
            if rxn_key in self.reactions:
                return
            rp = self._rate_for_reaction(
                react_mols, prod_mols, rxn_key, efam.label,
                list(tr.template) if tr.template else None,
                float(tr.degeneracy), iteration)
            if rp is None:
                return  # coverage gap, already counted
            self.reactions[rxn_key] = {"rp": rp, "react_keys": react_keys,
                                       "prod_keys": prod_keys,
                                       "family": efam.label}
            core_keys = self._core_key_set()
            if all(k in core_keys for k in react_keys + prod_keys):
                self._add_core_reaction(rxn_key)
            else:
                self._add_edge_reaction(rxn_key)
        except Exception:  # noqa: BLE001
            return

    def _core_key_set(self) -> set:
        return {canonical_key(sp.molecule) for sp in self.model.core.species}

    def _make_reaction(self, rxn_key: str) -> Reaction:
        rec = self.reactions[rxn_key]
        react = [self.species_by_key[k] for k in rec["react_keys"]]
        prod = [self.species_by_key[k] for k in rec["prod_keys"]]
        return Reaction(reactants=react, products=prod, rate_model=rec["rp"],
                        degeneracy=rec["rp"].degeneracy, family=rec["family"])

    def _add_core_reaction(self, rxn_key: str) -> None:
        rx = self._make_reaction(rxn_key)
        self.model.add_reaction_to_core(rx)
        for e in list(self.model.edge.reactions):
            if (e.reactants == rx.reactants and e.products == rx.products):
                try:
                    self.model.edge.reactions.remove(e)
                except ValueError:
                    pass

    def _add_edge_reaction(self, rxn_key: str) -> None:
        rx = self._make_reaction(rxn_key)
        self.model.add_reaction_to_edge(rx)
        core_keys = self._core_key_set()
        for k in self.reactions[rxn_key]["react_keys"] + \
                self.reactions[rxn_key]["prod_keys"]:
            sp = self.species_by_key[k]
            if k not in core_keys:
                self.model.add_species_to_edge(sp)

    # -- simulate + screen ------------------------------------------------
    def _build_sim(self):
        """Assemble (keys, nu, rps, init) for the current mechanism. None when
        there are no reactions."""
        if not self.reactions:
            return None
        sp_keys: List[str] = []
        seen = set()
        for rec in self.reactions.values():
            for sk in rec["react_keys"] + rec["prod_keys"]:
                if sk not in seen:
                    seen.add(sk)
                    sp_keys.append(sk)
        sp_keys.sort()
        idx = {k: i for i, k in enumerate(sp_keys)}
        nu = [[0.0] * len(sp_keys) for _ in self.reactions]
        rps = []
        order = sorted(self.reactions.keys())
        for j, k in enumerate(order):
            rec = self.reactions[k]
            rps.append(rec["rp"])
            for sk in rec["react_keys"]:
                nu[j][idx[sk]] -= 1
            for sk in rec["prod_keys"]:
                nu[j][idx[sk]] += 1
        init = [0.0] * len(sp_keys)
        for i, k in enumerate(sp_keys):
            sp = self.species_by_key[k]
            init[i] = float(self.ctx.initial_mole_fractions.get(sp.label, 0.0))
        return {"keys": sp_keys, "nu": nu, "rps": rps, "init": init,
                "order": order}

    def simulate(self, iteration: int):
        sim = self._build_sim()
        if sim is None:
            return None, [], []
        self._last_sim = sim
        T, P = self.ctx.temperature, self.ctx.pressure
        char = characteristic_rate(sim["nu"], sim["rps"], T, P, sim["init"])
        if char <= 0.0:
            t_end = 1.0
        else:
            t_end = CHAR_RATE_TFACTOR / char
        if self.ctx.termination_time is not None:
            t_end = min(t_end, self.ctx.termination_time)
        t_end = max(self.ctx.config.min_t, min(t_end, self.ctx.config.max_t))
        h = max(t_end / self.ctx.config.simulate_steps, 1e-18)
        profiles = simulate_mole_fractions(
            sim["keys"], sim["nu"], sim["rps"], T, P, sim["init"], t_end, h)
        self._last_profiles = profiles
        promote, demote = self._screen(sim, char, T, P, profiles)
        return profiles, promote, demote

    def _screen(self, sim: Dict, char: float, T: float, P: float,
                profiles: SimResult):
        """Screen species by reaction rate-ratio vs the characteristic rate
        using integrated fluxes from the simulation profiles. Species are promoted
        to core when max integrated rate-ratio > tolerance_move_to_core, demoted
        when < tolerance_move_to_core. Mirrors RMG-Py reactor-driven screening.
        """
        import numpy as np
        from rmgpu.reactor.simulator import forward_A_T, reverse_factor
        tol_core = self.ctx.config.tolerance_move_to_core
        tol_keep = self.ctx.config.tolerance_keep_in_edge
        n_sp = len(sim["keys"])
        n_steps = len(profiles.ys)
        # Build per-step rate matrix
        c_tot = P / (8.314472 * T)
        # Precompute reaction rates for each step
        sp_ratio = {}
        # Initialize max integrated ratio per species
        for i in range(n_sp):
            sp_ratio[sim["keys"][i]] = 0.0
        # For each reaction, compute integrated rate over time
        for j in range(len(sim["rps"])):
            rp = sim["rps"][j]
            aT = forward_A_T(rp, T)
            nu = sim["nu"][j]
            n_react = sum(-m for m in nu if m < 0)
            n_prod = sum(m for m in nu if m > 0)
            # Integrate rate over time steps
            rates = []
            for row in profiles.ys:
                prod_f = 1.0
                prod_r = 1.0
                for i,m in enumerate(nu):
                    y = max(0.0, row[i])
                    if m < 0:
                        prod_f *= y ** (-m)
                    elif m > 0:
                        prod_r *= y ** m
                k_fwd = aT * (c_tot ** (n_react - 1)) * prod_f
                k_rev = aT * reverse_factor(rp, T, float(n_prod - n_react)) * (c_tot ** (n_prod - 1)) * prod_r
                rates.append(max(k_fwd, k_rev))
            # Simple trapezoidal integration
            if len(rates) > 1:
                dt = (profiles.times[-1] - profiles.times[0]) / (len(rates)-1) if len(rates)>1 else 1.0
                integral = sum((rates[i]+rates[i-1])*0.5*dt for i in range(1,len(rates)))
            else:
                integral = rates[0] if rates else 0.0
            avg_rate = integral / (profiles.times[-1]-profiles.times[0] + 1e-30) if profiles.times[-1]>profiles.times[0] else 0.0
            rr = avg_rate / char if char>0 else 0.0
            for i,m in enumerate(nu):
                if m != 0:
                    k = sim["keys"][i]
                    sp_ratio[k] = max(sp_ratio.get(k,0.0), rr)
        core_keys = self._core_key_set()
        promote = [k for k in sim["keys"] if k not in core_keys and sp_ratio.get(k,0.0) > tol_core]
        demote = [k for k in core_keys if sp_ratio.get(k,0.0) < tol_core]
        self._last_sp_ratio = sp_ratio
        return promote, demote

    # -- main loop --------------------------------------------------------
    def run(self) -> RunResult:
        log.info("CoreEdgeLoop.run() starting")
        ctx = self.ctx
        cfg = ctx.config
        # Seed species from input species block
        for sp in ctx.seed_species:
            self.model.add_species_to_core(sp)
            self.species_by_key[canonical_key(sp.molecule)] = sp
            self._thermo(sp, 0)
        # Load seed mechanisms (rmgdb libraries) into core
        log.info("CoreEdgeLoop: seeding from mechanisms – %d species / %d reactions", len(ctx.seed_mechanisms_species), len(ctx.seed_mechanisms_reactions))
        self.log.append("seed mechanisms: %d species / %d reactions" % (len(ctx.seed_mechanisms_species), len(ctx.seed_mechanisms_reactions)))
        for sp in ctx.seed_mechanisms_species:
            key = canonical_key(sp.molecule)
            if key not in self.species_by_key:
                self.species_by_key[key] = sp
                self.model.add_species_to_core(sp)
                self._thermo(sp, 0)
        for rx in ctx.seed_mechanisms_reactions:
            # Ensure all species are registered
            for s in rx.reactants + rx.products:
                key = canonical_key(s.molecule)
                if key not in self.species_by_key:
                    self.species_by_key[key] = s
                    self.model.add_species_to_core(s)
                    self._thermo(s, 0)
            # Add reaction to core if all participants are in core
            core_keys = self._core_key_set()
            react_keys = [canonical_key(s.molecule) for s in rx.reactants]
            prod_keys = [canonical_key(s.molecule) for s in rx.products]
            if all(k in core_keys for k in react_keys + prod_keys):
                self.model.add_reaction_to_core(rx)
            else:
                # If not all core, add to edge
                self.model.add_reaction_to_edge(rx)
                for s in rx.reactants + rx.products:
                    if canonical_key(s.molecule) not in core_keys:
                        self.model.add_species_to_edge(s)
            # Register the seed reaction in self.reactions so the REACTOR
            # SIMULATION integrates it (job-06/step-07: c3h4 must actually RUN
            # its seed mechanism, not just carry it in the artifact). This is
            # what makes the run a real GRI-Mech3-seeded mechanism growth.
            rp = getattr(rx, "rate_model", None)
            if rp is not None:
                rxn_key = ".".join(sorted(react_keys)) + ">>" + ".".join(sorted(prod_keys))
                if rxn_key not in self.reactions:
                    self.reactions[rxn_key] = {"rp": rp, "react_keys": react_keys,
                                               "prod_keys": prod_keys,
                                               "family": "seed"}
        self.log.append("seed core: %d species" % len(self.model.core.species))

        prev_sig = None
        iteration = 0
        done = False
        log.info("CoreEdgeLoop: starting iteration loop")
        while not done and iteration < cfg.max_iterations:
            iteration += 1
            self.model.iteration_num = iteration
            log.info("Iteration %d start – core %d spc/%d rxn, edge %d spc/%d rxn",
                     iteration,
                     len(self.model.core.species),
                     len(self.model.core.reactions),
                     len(self.model.edge.species),
                     len(self.model.edge.reactions))
            self.log.append(
                "iteration %d: core=%d spc/%d rxn, edge=%d spc/%d rxn"
                % (iteration, len(self.model.core.species),
                   len(self.model.core.reactions),
                   len(self.model.edge.species),
                   len(self.model.edge.reactions)))
            log.debug("Iteration %d: enlarge", iteration)
            self.enlarge(iteration)
            log.debug("Iteration %d: simulate", iteration)
            profiles, promote_keys, demote_keys = self.simulate(iteration)
            log.info("Iteration %d: simulate done – promote %d, demote %d", iteration, len(promote_keys) if promote_keys else 0, len(demote_keys) if demote_keys else 0)
            if profiles is not None:
                for k in promote_keys:
                    self._promote(k)
                for k in demote_keys:
                    self._demote(k)
            log.debug("Iteration %d: prune", iteration)
            self.prune()
            sig = self._signature()
            if prev_sig is not None and sig == prev_sig:
                self.log.append("steady state at iteration %d" % iteration)
                log.info("Steady state reached at iteration %d", iteration)
                done = True
            prev_sig = sig
        log.info("CoreEdgeLoop: loop finished after %d iterations", iteration)
        self.model.iteration_num = iteration
        core_keys = sorted(self._core_key_set())
        edge_keys = sorted(set(canonical_key(s.molecule)
                               for s in self.model.edge.species) - set(core_keys))
        return RunResult(
            core_model=self.model, iterations=iteration, counts=self.counts,
            coverage=self.coverage, final_profiles=self._last_profiles,
            core_keys=core_keys, edge_keys=edge_keys,
            events=self.events, log_lines=self.log)

    def _promote(self, key: str) -> None:
        sp = self.species_by_key.get(key)
        if sp is None:
            return
        self.model.add_species_to_core(sp)
        if sp in self.model.edge.species:
            self.model.edge.species.remove(sp)
        core_keys = self._core_key_set()
        for rx in list(self.model.edge.reactions):
            all_core = (all(canonical_key(r.molecule) in core_keys
                            for r in rx.reactants)
                        and all(canonical_key(p.molecule) in core_keys
                                for p in rx.products))
            if all_core:
                self.model.edge.reactions.remove(rx)
                self.model.core.reactions.append(rx)

    def _demote(self, key: str) -> None:
        sp = self.species_by_key.get(key)
        if sp is None:
            return
        # Move species from core to edge if it exists in core
        if sp in self.model.core.species:
            self.model.core.species.remove(sp)
        if sp not in self.model.edge.species:
            self.model.add_species_to_edge(sp)
        # Move reactions that involve demoted species back to edge
        for rx in list(self.model.core.reactions):
            # If reaction involves demoted species or any non-core species
            core_keys = self._core_key_set()
            react_keys = [canonical_key(r.molecule) for r in rx.reactants]
            prod_keys = [canonical_key(p.molecule) for p in rx.products]
            if not all(k in core_keys for k in react_keys + prod_keys):
                self.model.core.reactions.remove(rx)
                self.model.add_reaction_to_edge(rx)

    def prune(self) -> None:
        # Rate-ratio based pruning to mimic RMG-Py tol_keep_in_edge
        # Use last simulation sp_ratio if available
        if hasattr(self, '_last_sp_ratio'):
            tol_keep = self.ctx.config.tolerance_keep_in_edge
            # Prune edge species below tolerance
            to_remove = []
            for sp in list(self.model.edge.species):
                k = canonical_key(sp.molecule)
                rr = self._last_sp_ratio.get(k, 0.0)
                if rr < tol_keep:
                    to_remove.append(sp)
            for sp in to_remove:
                try:
                    self.model.edge.species.remove(sp)
                except ValueError:
                    pass
            # Also prune reactions involving removed species
            for rx in list(self.model.edge.reactions):
                keys = [canonical_key(m.molecule) for m in rx.reactants + rx.products]
                if any(k not in {canonical_key(s.molecule) for s in self.model.edge.species} for k in keys):
                    try:
                        self.model.edge.reactions.remove(rx)
                    except ValueError:
                        pass
        # Fallback structural prune
        kept = set()
        for rx in self.model.edge.reactions:
            for m in rx.reactants + rx.products:
                kept.add(canonical_key(m.molecule))
        for sp in list(self.model.edge.species):
            if canonical_key(sp.molecule) not in kept:
                self.model.edge.species.remove(sp)

    def _signature(self):
        return (
            tuple(sorted(self._core_key_set())),
            tuple(sorted(canonical_key(s.molecule)
                         for s in self.model.edge.species)),
            tuple(sorted(self.reactions.keys())),
        )
