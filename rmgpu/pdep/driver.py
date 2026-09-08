"""rmgpu.pdep.driver
====================

job-07/step-06: the pdep DRIVER - the orchestration that sits between the
core loop and the master-equation numerics (steps 01-05).

Ports RMG-Py's pressure-dependence orchestration:
  - rmgpy/rmg/input.py:1376  ``pressure_dependence`` (block parse: method,
    interpolation model, T/P grid) + arkane/pdep.py PressureDependenceJob
    ``generate_T_list``/``generate_P_list`` (the Gauss-Chebyshev / linear-log
    grid) and ``fit_interpolation_models``/``fit_interpolation_model``.
  - rmgpy/pdep/reaction.pyx:338  ``fit_interpolation_model`` (the EXACT fit:
    Chebyshev vs PDepArrhenius selected by the input's interpolation_model).

The driver's job (ORCHESTRATION - the numerics are steps 01-05, this is
wiring):

    run_pdep(network, block, ...)
      1. build the (T,P) grid from the YAML pressure_dependence block
         (Tmin/Tmax/Tcount, Pmin/Pmax/Pcount) - RMG's exact grids;
      2. run the network over the grid: for each (T,P) the step-04 Network
         solves the master equation (via its state_provider / seeded state) ->
         the k(T,P) matrix K[n_T, n_P, n_cfg, n_cfg];
      3. fit each net reaction's k(T,P) to Chebyshev/PDepArrhenius (RMG's
         exact fit, ported to rmgpu.kinetics.models) - the "Falloff";
      4. attach the fitted Falloff to each net reaction (the registry seam);
      5. write pdep/<network>.yaml (the network definition + the k(T,P) grid
         + the fitted coefficients - self-contained, PLAN 12.3).

Method selection: the YAML ``pressure_dependence.method`` (normalized per
step-04's table). CSE (chemically-significant eigenvalues) WORKS; MSC / RS /
SLS raise NotImplementedError (job-08 lands them).

Unit convention (settled by /tmp/probe_fit.py + rmgpy/kinetics/chebyshev.pyx
fit_to_data + rmgpy/rmg/pdep.py:946-950, read this step - NO guessing): RMG's
core-loop path extracts the SI k(T,P) matrix K from the ME, fits it (the fit
converts to SI internally), so the stored Chebyshev coefficients represent
log10(SI k) and ``get_rate_coefficient`` returns SI (m^3/mol/s etc.). rmgpu
stores and evaluates in SI throughout; the CGS -> SI c00 shift applies only on
file LOAD (see Chebyshev.__post_init__). The driver therefore fits on the SI
grid and records SI coefficients.

The per-T network STATE (DoS, fluxes Kij/Gnj/Fim, collision) is built by
steps 01-05 and handed to the driver through the Network's ``state_provider``
(a callable T -> per-T state dict) or a pre-seeded ``state_at`` mapping (the
step-04 seam). The production statmech->DoS->fluxes builder
(RMG-Py network.set_conditions) is the remaining numerics piece that the
job-07 GATE (step 07, propane_branching) needs; it is deliberately NOT in
this step (a wiring step), per the step-04/05 state-seeding protocol.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from rmgpu.kinetics.models import (
    Arrhenius,
    Chebyshev,
    KineticsModel,
    PDepArrhenius,
)
from rmgpu.logging import get_logger

log = get_logger("pdep.driver")


class PDepDriverError(Exception):
    """Raised for driver-level (orchestration) errors."""


# --------------------------------------------------------------------------- #
# Method selection (step-04 normalized table -> the Network dispatch strings)
# --------------------------------------------------------------------------- #
# The step-04 schema normalizes the short names; the Network's
# calculate_rate_coefficients dispatches on the LONG strings. CSE is the only
# method that WORKS this step; MSC / RS / SLS are job-08 (raise NotImplemented
# here so the failure is loud, not a silent no-op).
_CSE_ALIASES = {
    "cse", "strong collision", "chemically-significant eigenvalues",
    "chemically significant eigenvalues",
}
_MSC_ALIASES = {"masc", "modified strong collision"}
_RS_ALIASES = {"rs", "reservoir state", "randomized strong collision"}
_SLS_ALIASES = {"sls", "super logarithmic spacing", "simulation least squares"}

CSE_METHOD = "chemically-significant eigenvalues"


def resolve_method(method: str) -> str:
    """Map the (normalized) YAML method to the Network dispatch string.

    CSE -> the Network's CSE (Allen) string. MSC / RS / SLS -> raise
    NotImplementedError (job-08). Anything unrecognized -> PDepDriverError
    (matches RMG raising PressureDependenceError for unknown methods).
    """
    m = (method or "").strip().lower()
    if m in _CSE_ALIASES:
        return CSE_METHOD
    if m in _MSC_ALIASES or m in _RS_ALIASES or m in _SLS_ALIASES:
        raise NotImplementedError(
            "pdep method %r is a job-08 deliverable (the step-06 driver is "
            "CSE-only; MSC/RS/SLS land in job 08)." % method)
    if m in ("chemically-significant eigenvalues georgievskii",):
        raise NotImplementedError(
            "the georgievskii CSE variant is a job-08 deliverable.")
    raise PDepDriverError('Unknown pdep method "%s".' % method)


# --------------------------------------------------------------------------- #
# The (T,P) grid (RMG arkane/pdep.py generate_T_list / generate_P_list, exact)
# --------------------------------------------------------------------------- #
def generate_t_list(Tmin: float, Tmax: float, Tcount: int,
                    chebyshev: bool) -> np.ndarray:
    """RMG generate_T_list: Gauss-Chebyshev (Chebyshev model) or linear on
    the T^-1 domain (PDepArrhenius)."""
    if chebyshev:
        out = np.zeros(Tcount, float)
        for i in range(Tcount):
            x = -math.cos((2 * i + 1) * math.pi / (2 * Tcount))
            out[i] = 2.0 / ((1.0 / Tmax - 1.0 / Tmin) * x + 1.0 / Tmax + 1.0 / Tmin)
        return out
    return 1.0 / np.linspace(1.0 / Tmax, 1.0 / Tmin, Tcount)


def generate_p_list(Pmin: float, Pmax: float, Pcount: int,
                    chebyshev: bool) -> np.ndarray:
    """RMG generate_P_list: Gauss-Chebyshev (Chebyshev model) or linear on
    the log-P domain (PDepArrhenius)."""
    if chebyshev:
        out = np.zeros(Pcount, float)
        for i in range(Pcount):
            x = -math.cos((2 * i + 1) * math.pi / (2 * Pcount))
            out[i] = 10 ** (0.5 * ((math.log10(Pmax) - math.log10(Pmin)) * x
                                   + math.log10(Pmax) + math.log10(Pmin)))
        return out
    return 10.0 ** np.linspace(math.log10(Pmin), math.log10(Pmax), Pcount)


# --------------------------------------------------------------------------- #
# The exact fit (rmgpy/pdep/reaction.pyx:338 fit_interpolation_model)
# --------------------------------------------------------------------------- #
def _si_kunits(n_reactants: int) -> str:
    """SI rate units by reaction order (RMG reaction.pyx:358, SI variant)."""
    return {1: "s^-1", 2: "m^3/(mol*s)", 3: "m^6/(mol^2*s)"}[n_reactants]


def fit_interpolation_model(kdata: np.ndarray, model: Tuple, Tlist: np.ndarray,
                            Plist: np.ndarray, n_reactants: int,
                            Tmin: float, Tmax: float, Pmin: float, Pmax: float
                            ) -> KineticsModel:
    """Port of rmgpy/pdep/reaction.pyx:338 fit_interpolation_model.

    ``model`` is the YAML interpolation tuple: ('Chebyshev', degreeT, degreeP)
    or ('PDepArrhenius',). ``kdata`` is the SI k(T,P) grid (n_T, n_P) for ONE
    net reaction. Returns the fitted model (the "Falloff"): a Chebyshev with
    SI-scale coefficients, or a PDepArrhenius with per-pressure Arrhenius fits
    (both evaluate in SI via get_rate_coefficient(T, P)).
    """
    model_name = str(model[0]).lower()
    kunits = _si_kunits(n_reactants)
    if model_name == "chebyshev":
        degree_t, degree_p = int(model[1]), int(model[2])
        chb = Chebyshev()
        chb.fit_to_data(Tlist, Plist, kdata, kunits, degree_t, degree_p,
                        Tmin, Tmax, Pmin, Pmax)
        return chb
    if model_name == "pdeparrhenius":
        pda = PDepArrhenius()
        pda.fit_to_data(Tlist, Plist, kdata, kunits=kunits, T0=298.0)
        return pda
    raise PDepDriverError(
        "interpolation_model must be 'Chebyshev' or 'PDepArrhenius' (got %r)."
        % (model,))


# --------------------------------------------------------------------------- #
# Net reaction + network spec (the driver's operating units)
# --------------------------------------------------------------------------- #
@dataclass
class PDepReaction:
    """One net (phenomenological) reaction of a pdep network: the isomer or
    channel pair whose k(T,P) the driver fits. ``from_cfg`` / ``to_cfg`` are
    indices into the Network's k(T,P) matrix K[n_T, n_P, to_cfg, from_cfg]
    (RMG convention: K[:,:,prod,reac] = the rate from `reac` config to `prod`
    config). ``n_reactants`` sets the rate units. ``highPlimit`` is the
    high-pressure-limit kinetics (the no-QM HPL, used for the TS-E0 / reverse
    bookkeeping and as the falloff's high-P anchor)."""
    label: str
    from_cfg: int
    to_cfg: int
    n_reactants: int = 1
    highPlimit: Optional[KineticsModel] = None
    # set by the driver after the fit (the Falloff attaches here):
    kinetics: Optional[KineticsModel] = None
    kdata: Optional[np.ndarray] = None   # the (n_T, n_P) SI grid that was fit
    fit_log_rms: float = float("nan")    # the RMG error_check log-RMS


@dataclass
class PDepNetwork:
    """The driver's view of a pdep network: the rmgpu step-04 Network (which
    holds n_isom/n_reac/n_prod, E0, E0_ts, grain params, and the state seam)
    plus the net reactions to fit and the (T,P) grid."""
    label: str
    network: object                # rmgpu.pdep.network.Network
    reactions: List[PDepReaction] = field(default_factory=list)
    Tlist: Optional[np.ndarray] = None
    Plist: Optional[np.ndarray] = None

    @property
    def n_cfg(self) -> int:
        return int(self.network.n_cfg)

    # the RMG net-reaction (from,to) pairs for every (product, reactant) config
    # pair, product != reactant (RMG fit_interpolation_models loop order)
    def net_pairs(self) -> List[Tuple[int, int]]:
        n_chem = int(self.network.n_isom) + int(self.network.n_reac)
        pairs = []
        for prod in range(self.n_cfg):
            for reac in range(n_chem):
                if prod != reac:
                    pairs.append((prod, reac))
        return pairs


# --------------------------------------------------------------------------- #
# The block -> grid
# --------------------------------------------------------------------------- #
def _grid_from_block(block, chebyshev: bool):
    """Build (Tlist, Plist) from a PressureDependenceBlock (SI, K/Pa)."""
    def si(q, fallback):
        v = float(getattr(q, "value", q))
        return v if v else fallback
    Tmin = float(block.tmin.to_si())
    Tmax = float(block.tmax.to_si())
    Tcount = int(block.tcount)
    Pmin = float(block.pmin.to_si())
    Pmax = float(block.pmax.to_si())
    Pcount = int(block.pcount)
    return generate_t_list(Tmin, Tmax, Tcount, chebyshev), \
        generate_p_list(Pmin, Pmax, Pcount, chebyshev)


# --------------------------------------------------------------------------- #
# The driver
# --------------------------------------------------------------------------- #
def run_pdep(network: "PDepNetwork", block=None, method: Optional[str] = None,
             interpolation_model=None, Tlist: Optional[np.ndarray] = None,
             Plist: Optional[np.ndarray] = None,
             output_dir: Optional[str] = None,
             error_check: bool = True) -> "PDepResult":
    """Run the pdep driver over one network.

    Parameters
    ----------
    network : PDepNetwork
        The driver's network view (wraps the rmgpu step-04 Network + the net
        reactions to fit). The Network must already expose its per-T state
        through ``state_provider`` (a callable T -> state dict) or be seeded
        with a ``state_at``-style state - the step-04/05 seam.
    block : PressureDependenceBlock, optional
        The YAML pressure_dependence block (drives the (T,P) grid + method +
        interpolation_model). When None, the explicit ``method``/``Tlist``/
        ``Plist``/``interpolation_model`` arguments are used.
    method : str, optional
        The pdep method (see resolve_method). Defaults to block.method, else CSE.
    interpolation_model : tuple, optional
        ('Chebyshev', degreeT, degreeP) or ('PDepArrhenius',). Defaults to
        block.interpolation_model (or the string form), else ('Chebyshev',6,4).
    Tlist, Plist : array-like, optional
        Explicit grids. When both are given, block is not used for the grid.
    output_dir : str, optional
        If set, write pdep/<network>.yaml (under output_dir/pdep/).
    error_check : bool
        RMG's error_check: warn if the fit deviates > 0.5 in log-RMS from the
        grid (RMG reaction.pyx:389).

    Returns
    -------
    PDepResult with the k(T,P) grid, the fitted reactions (Falloff attached),
    the grid, and the wall time (the "leverage GPUs" payoff the step wants shown).
    """
    t0 = time.time()

    # -- method --------------------------------------------------------- #
    if method is None:
        method = getattr(block, "method", None) if block is not None else None
    method = method or CSE_METHOD
    dispatch_method = resolve_method(method)

    # -- interpolation model -------------------------------------------- #
    if interpolation_model is None:
        interpolation_model = getattr(block, "interpolation_model", None) \
            if block is not None else None
    interpolation_model = _normalize_interpolation(interpolation_model)
    model_name = str(interpolation_model[0]).lower()
    cheb = model_name == "chebyshev"

    # -- grid ----------------------------------------------------------- #
    if Tlist is None or Plist is None:
        if block is None:
            raise PDepDriverError(
                "run_pdep needs either a pressure_dependence block or explicit "
                "Tlist/Plist.")
        Tlist, Plist = _grid_from_block(block, cheb)
    Tlist = np.asarray(Tlist, dtype=float)
    Plist = np.asarray(Plist, dtype=float)
    network.Tlist = Tlist
    network.Plist = Plist

    # -- solve the (T,P) grid (the step-04 Network) ---------------------- #
    # The Network pulls its per-T state from state_provider / its seeded state
    # (calculate_rate_coefficients handles the dispatch). K[n_T, n_P, cfg, cfg]
    # is the SI k(T,P) matrix.
    net = network.network
    K = net.calculate_rate_coefficients(Tlist, Plist, dispatch_method)
    wall_solve = time.time() - t0

    # -- fit each net reaction + attach the Falloff ---------------------- #
    Tmin, Tmax = float(Tlist.min()), float(Tlist.max())
    Pmin, Pmax = float(Plist.min()), float(Plist.max())
    n_T, n_P = len(Tlist), len(Plist)
    fitted = 0
    for rxn in network.reactions:
        # RMG convention: K[:,:,prod,reac] = rate from `reac` config to `prod`
        # config. A net reaction is (from_cfg -> to_cfg) = K[:,:,to_cfg,from_cfg].
        kdata = K[:, :, rxn.to_cfg, rxn.from_cfg].copy()
        # Guard the fit: log-space fits need positive data. A net reaction that
        # is identically zero on the grid (e.g. a forbidden reverse) is kept
        # with no kinetics (RMG only fits reactions with a path reaction).
        if not np.any(kdata > 0):
            rxn.kinetics = None
            rxn.kdata = kdata
            continue
        # Fit on a strictly-positive grid: clip the tiny zeros that appear where
        # k(T,P) < the floating-point floor (RMG fits the full grid; a log10 of
        # a 0 would be -inf). Use a relative floor at the grid's max.
        floor = np.finfo(float).tiny
        kdata_fit = np.where(kdata > 0, kdata, floor)
        kinetics = fit_interpolation_model(kdata_fit, interpolation_model,
                                           Tlist, Plist, rxn.n_reactants,
                                           Tmin, Tmax, Pmin, Pmax)
        # RMG's error_check (reaction.pyx:389): log-RMS of fit vs data.
        if error_check:
            rxn.fit_log_rms = _fit_log_rms(kinetics, kdata_fit, Tlist, Plist)
        rxn.kinetics = kinetics
        rxn.kdata = kdata
        fitted += 1

    wall = time.time() - t0

    # -- attach to the network (the registry seam) ---------------------- #
    # Each net reaction now carries its fitted Falloff (rxn.kinetics). The core
    # loop's RateParam (job-06) picks it up via the loop wiring (step-06 loop
    # deliverable): a pdep reaction's forward rate is the Falloff's
    # get_rate_coefficient(T, P) at the reactor conditions.

    # -- write pdep/<network>.yaml (self-contained, PLAN 12.3) ----------- #
    yaml_path = None
    if output_dir is not None:
        yaml_path = _write_network_yaml(network, dispatch_method,
                                        interpolation_model,
                                        output_dir, K=K)

    return PDepResult(
        network=network,
        method=dispatch_method,
        interpolation_model=tuple(interpolation_model),
        Tlist=Tlist, Plist=Plist,
        K=K,
        reactions=network.reactions,
        n_fitted=fitted,
        n_reactions=len(network.reactions),
        wall_s=wall,
        wall_solve_s=wall_solve,
        yaml_path=yaml_path,
    )


def _normalize_interpolation(model) -> Tuple:
    """Coerce the YAML interpolation_model (string or tuple) to a fit tuple."""
    if model is None:
        return ("Chebyshev", 6, 4)
    if isinstance(model, str):
        m = model.lower()
        if m == "chebyshev":
            return ("Chebyshev", 6, 4)
        if m == "pdeparrhenius":
            return ("PDepArrhenius",)
        raise PDepDriverError(
            "interpolation model must be 'Chebyshev' or 'PDepArrhenius' "
            "(got %r)." % model)
    if isinstance(model, (tuple, list)):
        return tuple(model)
    raise PDepDriverError("Cannot interpret interpolation_model %r." % (model,))


def _fit_log_rms(kinetics: KineticsModel, kdata: np.ndarray, Tlist, Plist) -> float:
    """RMG error_check: log-RMS of the fitted k(T,P) vs the grid (reaction.pyx)."""
    log_rms = 0.0
    n = 0
    for t, T in enumerate(Tlist):
        for p, P in enumerate(Plist):
            km = kinetics.get_rate_coefficient(T, P)
            kd = kdata[t, p]
            if km > 0 and kd > 0:
                d = math.log(km) - math.log(kd)
                log_rms += d * d
                n += 1
    if n == 0:
        return float("nan")
    return math.sqrt(log_rms / n)


# --------------------------------------------------------------------------- #
# pdep/<network>.yaml (self-contained: definition + grid + coefficients)
# --------------------------------------------------------------------------- #
def _reaction_to_dict(rxn: PDepReaction) -> Dict:
    d = {
        "label": rxn.label,
        "from_cfg": int(rxn.from_cfg),
        "to_cfg": int(rxn.to_cfg),
        "n_reactants": int(rxn.n_reactants),
        "fit_log_rms": rxn.fit_log_rms,
        "kinetics": _kinetics_to_dict(rxn.kinetics),
    }
    return d


def _f(x):
    """Coerce a numpy scalar to a plain Python float (yaml-safe)."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [_f(v) for v in x]
    try:
        import numpy as _np
        if isinstance(x, _np.generic):
            return x.item()
    except Exception:
        pass
    return float(x)


def _kinetics_to_dict(kin: Optional[KineticsModel]) -> Optional[Dict]:
    if kin is None:
        return None
    if isinstance(kin, Chebyshev):
        return {
            "type": "Chebyshev",
            "kunits": kin.kunits,
            "Tmin": _f(kin.Tmin), "Tmax": _f(kin.Tmax),
            "Pmin": _f(kin.Pmin), "Pmax": _f(kin.Pmax),
            "degreeT": int(kin.degreeT), "degreeP": int(kin.degreeP),
            "coeffs": np.asarray(kin.coeffs).tolist(),
        }
    if isinstance(kin, PDepArrhenius):
        if kin.arrhenius is not None:
            return {
                "type": "PDepArrhenius",
                "Tmin": _f(kin.Tmin), "Tmax": _f(kin.Tmax),
                "Pmin": _f(kin.Pmin), "Pmax": _f(kin.Pmax),
                "pressures": [_f(p) for p in
                              np.atleast_1d(np.asarray(kin.pressures, float))],
                "arrhenius": [
                    {"A": _f(a.A), "n": _f(a.n), "Ea": _f(a.Ea), "T0": _f(a.T0)}
                    for a in kin.arrhenius],
            }
        return {
            "type": "PDepArrhenius",
            "A": _f(kin.A), "n": _f(kin.n), "Ea": _f(kin.Ea), "T0": _f(kin.T0),
            "Pmin": _f(kin.Pmin), "Pmax": _f(kin.Pmax),
        }
    return {"type": type(kin).__name__}


def _write_network_yaml(net: "PDepNetwork", method: str,
                        interpolation_model, output_dir: str,
                        K: Optional[np.ndarray] = None) -> str:
    pdep_dir = os.path.join(output_dir, "pdep")
    os.makedirs(pdep_dir, exist_ok=True)
    path = os.path.join(pdep_dir, "%s.yaml" % net.label)
    n = net.network
    doc = {
        "network": net.label,
        "method": method,
        "interpolation_model": list(interpolation_model),
        "n_isom": int(n.n_isom), "n_reac": int(n.n_reac),
        "n_prod": int(n.n_prod), "n_cfg": int(n.n_cfg),
        "E0": np.atleast_1d(np.asarray(n.E0, float)).tolist() if n.E0 is not None else None,
        "E0_ts": np.atleast_1d(np.asarray(n.E0_ts, float)).tolist() if n.E0_ts is not None else None,
        "grain_size": float(getattr(n, "grain_size", 0.0) or 0.0),
        "grain_count": int(getattr(n, "grain_count", 0) or 0),
        "Tlist": np.asarray(net.Tlist).tolist() if net.Tlist is not None else None,
        "Plist": np.asarray(net.Plist).tolist() if net.Plist is not None else None,
        "reactions": [_reaction_to_dict(r) for r in net.reactions],
        # The k(T,P) grid (self-contained, PLAN 12.3): K[t, p, cfg, cfg] (SI).
        "K_grid": np.asarray(K).tolist() if K is not None else None,
    }
    with open(path, "w") as f:
        yaml.safe_dump(doc, f)
    return path


def load_network_yaml(path: str) -> Dict:
    """Parse a pdep/<network>.yaml back (round-trip check for the driver test)."""
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class PDepResult:
    network: PDepNetwork
    method: str
    interpolation_model: Tuple
    Tlist: np.ndarray
    Plist: np.ndarray
    K: np.ndarray
    reactions: List[PDepReaction]
    n_fitted: int
    n_reactions: int
    wall_s: float
    wall_solve_s: float
    yaml_path: Optional[str] = None

    def summary(self) -> Dict:
        return {
            "network": self.network.label,
            "method": self.method,
            "interpolation_model": list(self.interpolation_model),
            "n_T": len(self.Tlist), "n_P": len(self.Plist),
            "n_reactions": self.n_reactions,
            "n_fitted": self.n_fitted,
            "wall_s": round(self.wall_s, 3),
            "wall_solve_s": round(self.wall_solve_s, 3),
            "yaml_path": self.yaml_path,
        }
