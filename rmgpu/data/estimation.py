"""Estimation resolvers: library -> ML -> MLCoverageError.

These two functions are the ONLY estimation code in rmgpu. They replace
RMG-Py's multi-method fallback chains (PLAN.md 3, 14): a library hit wins,
otherwise the ML estimator is used, and if neither covers the structure a
``MLCoverageError`` is raised and the coverage counter is incremented - the
error is NEVER swallowed into a third method. There is no group additivity,
no rate rules, and no other branch (PLAN.md 14 anti-goals).

Both resolvers are instrumented with an :class:`EstimationCounts` object so
the thesis test (job-04 step-06, gate note 3) can prove the no-fallback
invariant: library hits + ML hits + coverage errors == total attempts, and
the split is reported.

Unit convention: everything returned is SI (J, K, Pa, mol). Library entries
are stored in rmgdb in CGS (kcal/mol, cal/(mol*K)); the resolvers convert
on the way out. The ``ml`` estimators are threaded in as an argument (not a
global) - job 06's driver constructs the estimators once and passes them
down, and these functions stay pure with respect to that.

Reaction SMILES convention: atom-mapped reactants>>products, exactly as
``rmgpu.ml.kinetics_estimator.KineticsML`` and ``models/predict.py`` expect
(the RIGR featurizer requires atom-mapping). ``reaction_smiles`` on a
reaction dict is the authoritative input; when absent the resolver builds
unmapped ``reactants >> products`` from the species structures (a valid
fallback for lookup purposes - PLAN.md 3b notes the atom correspondence
comes from the recipe engine, job-05).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from rmgpu.data.entries import ThermoEntry
from rmgpu.data.thermo import NASA, Wilhoit
from rmgpu.kinetics.models import Arrhenius, KineticsModel
from rmgpu.ml.thermo_estimator import MLCoverageError, ThermoPrediction
from rmgpu.molecule.molecule import Molecule

__all__ = [
    "Cal",
    "KCAL",
    "CpGrid",
    "EstimationCounts",
    "estimate_thermo",
    "estimate_kinetics",
    "species_to_smiles",
    "reaction_to_smiles",
]

# rmgdb unit factors (library files are CGS; rmgpu is SI)
KCAL = 4184.0  # J/mol per kcal/mol
Cal = 4.184  # J/(mol*K) per cal/(mol*K)

# The Cp grid the thermo checkpoint predicts at (PLAN.md 3b).
CpGrid = (300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0)


@dataclass
class EstimationCounts:
    """Instrumentation counters for the estimation resolvers.

    The thesis test reads these to prove no fallback ever happens: the sum
    of the three fields must equal the number of resolver attempts, and the
    report shows the split (gate note 3).
    """

    library_hits: int = 0
    ml_hits: int = 0
    coverage_errors: int = 0

    @property
    def total(self) -> int:
        return self.library_hits + self.ml_hits + self.coverage_errors

    def as_dict(self) -> dict:
        return {
            "library_hits": self.library_hits,
            "ml_hits": self.ml_hits,
            "coverage_errors": self.coverage_errors,
            "total": self.total,
        }


# ---------------------------------------------------------------------------
# Unit conversion (rmgdb stores CGS; rmgpu is SI)
# ---------------------------------------------------------------------------


def convert_energy(value: float, unit: str) -> float:
    """Convert an energy value to J/mol."""
    if value is None:
        raise MLCoverageError("stored energy is None")
    v = float(value)
    if v != v:
        raise MLCoverageError("stored energy is NaN (NULL in rmgdb)")
    u = str(unit).strip()
    if u in ("kcal/mol",):
        return v * KCAL
    if u in ("cal/mol",):
        return v * Cal
    if u in ("J/mol",):
        return v
    if u in ("kJ/mol",):
        return v * 1.0e3
    if u in ("kK", "K"):  # energy expressed as temperature (R*T)
        return v * 8.314472
    raise MLCoverageError(f"unknown energy unit {unit!r}")


def convert_entropy(value: float, unit: str) -> float:
    """Convert an entropy value to J/(mol*K)."""
    if value is None:
        raise MLCoverageError("stored entropy is None")
    v = float(value)
    if v != v:
        raise MLCoverageError("stored entropy is NaN (NULL in rmgdb)")
    u = str(unit).strip()
    if u in ("cal/(mol*K)",):
        return v * Cal
    if u in ("J/(mol*K)",):
        return v
    if u in ("kcal/(mol*K)",):
        return v * KCAL
    if u in ("dimensionless",):
        return v * 8.314472
    raise MLCoverageError(f"unknown entropy unit {unit!r}")


def convert_cp_grid(tdata: list, cdata: list, unit: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert a Tdata/Cpdata pair to SI arrays (K, J/(mol*K))."""
    u = str(unit).strip()
    if u in ("cal/(mol*K)",):
        factor = Cal
    elif u in ("J/(mol*K)",):
        factor = 1.0
    elif u in ("kcal/(mol*K)",):
        factor = KCAL
    else:
        raise MLCoverageError(f"unknown Cp unit {unit!r}")
    T = np.asarray(tdata, dtype=np.float64)
    Cp = np.asarray(cdata, dtype=np.float64) * factor
    return T, Cp


# ---------------------------------------------------------------------------
# Cp model for library hits (Wilhoit fit over the 7-point grid)
# ---------------------------------------------------------------------------


def _wilhoit_from_grid(T: np.ndarray, Cp: np.ndarray) -> tuple[Wilhoit, float]:
    """Fit a Wilhoit model to a Cp grid (RMG fits the same 7-point grid).

    Seven free parameters (Cp0, CpInf, a0..a3, B) over seven grid points, so
    the fit is (near-)exact when it converges; scipy least-squares with a
    bounded fallback to the estimator's coarse fit (Cp0=Cp[0], CpInf=Cp[-1],
    a_i=0, B=500). Returns (model, max_abs_fit_error_J_molK).
    """
    T = np.asarray(T, dtype=np.float64)
    Cp = np.asarray(Cp, dtype=np.float64)

    def wilhoit_cp(p, t):
        Cp0, CpInf, a0, a1, a2, a3, B = p
        y = t / (t + B)
        return Cp0 + (CpInf - Cp0) * y * y * (
            1 + (y - 1) * (a0 + y * (a1 + y * (a2 + y * a3)))
        )

    def _coarse():
        from rmgpu.ml.thermo_estimator import _fit_wilhoit as _est_fit

        m, err = _est_fit(T, Cp)
        return (m.Cp0, m.CpInf, m.a0, m.a1, m.a2, m.a3, m.B), float(err)

    def _fit_from(p0, a_bound, b_hi, nfev=5000):
        from scipy.optimize import least_squares

        res = least_squares(
            lambda p: wilhoit_cp(p, T) - Cp,
            p0,
            bounds=([-np.inf, -np.inf, -a_bound, -a_bound, -a_bound, -a_bound, 1.0],
                    [np.inf, np.inf, a_bound, a_bound, a_bound, a_bound, b_hi]),
            max_nfev=nfev,
        )
        err = float(np.max(np.abs(wilhoit_cp(res.x, T) - Cp)))
        return res.x, err

    best = None
    try:
        # Multi-start: the 7-parameter fit is multimodal; try several B
        # inits + bound widths and keep the smallest max residual.
        for B0, a_bound, b_hi in (
            (500.0, 10.0, 5000.0),
            (1000.0, 1.0e4, 2.0e4),
            (200.0, 1.0e4, 2.0e4),
            (2000.0, 1.0e4, 2.0e4),
        ):
            p0 = [Cp[0], Cp[-1], 0.0, 0.0, 0.0, 0.0, B0]
            x, err = _fit_from(p0, a_bound, b_hi)
            if best is None or err < best[1]:
                best = (x, err)
    except Exception:
        pass

    if best is None:
        params, err = _coarse()
    else:
        params, err = best

    Cp0, CpInf, a0, a1, a2, a3, B = params
    model = Wilhoit(
        Cp0=float(Cp0), CpInf=float(CpInf),
        a0=float(a0), a1=float(a1), a2=float(a2), a3=float(a3),
        B=float(B), Tmin=float(T[0]), Tmax=float(T[-1]),
    )
    err = float(np.max(np.abs(model.get_heat_capacity(T) - Cp)))
    return model, err


def _library_cp_model(entry: ThermoEntry) -> Any:
    """Build the Cp(T) representation for a library entry.

    - Tdata/Cpdata present -> interpolate the grid, fit a Wilhoit on top
      (the ML estimator wraps its 7-point prediction the same way).
    - NASA only          -> the NASA polynomials (Cp is exact).
    """
    from rmgpu.ml.thermo_estimator import CpModel

    if entry.Tdata and len(entry.Cpdata) == len(entry.Tdata):
        T, Cp = convert_cp_grid(entry.Tdata, entry.Cpdata, entry.Cpdata_unit)
        wilhoit, _ = _wilhoit_from_grid(T, Cp)
        # Store on the data-layer NASA/Wilhoit classes; CpModel (the
        # estimator's interpolator) expects a .get_heat_capacity(T).
        return CpModel(T=T, Cp=Cp, wilhoit=wilhoit)
    if entry.nasa_polynomials:
        return NASA(polynomials=entry.nasa_polynomials)
    raise MLCoverageError(
        f"library entry {entry.label!r} carries no Cp model (no Tdata/Cpdata, "
        "no NASA polynomials)"
    )


def _num(value) -> Optional[float]:
    """A stored numeric field: NaN (rmgdb's NULL via pandas) counts as absent."""
    if value is None:
        return None
    v = float(value)
    if v != v:  # NaN
        return None
    return v


def _library_H298_S298(entry: ThermoEntry) -> tuple[float, float]:
    """Hf298 / S298 (SI) for a library entry, from whichever model it has.

    rmgdb leaves H298/S298 NULL (-> NaN) for NASA-only entries (e.g. N2 in
    primaryThermoLibrary); those take their values from the NASA model,
    exactly as RMG-Py would.
    """
    H298 = _num(entry.H298)
    if H298 is not None:
        H298_si = convert_energy(H298, entry.H298_unit)
    elif entry.nasa_polynomials:
        H298_si = NASA(polynomials=entry.nasa_polynomials).get_enthalpy(298.15)
    else:
        raise MLCoverageError(
            f"library entry {entry.label!r} has no H298 and no NASA model"
        )
    S298 = _num(entry.S298)
    if S298 is not None:
        S298_si = convert_entropy(S298, entry.S298_unit)
    elif entry.nasa_polynomials:
        S298_si = NASA(polynomials=entry.nasa_polynomials).get_entropy(298.15)
    else:
        raise MLCoverageError(
            f"library entry {entry.label!r} has no S298 and no NASA model"
        )
    return H298_si, S298_si


# ---------------------------------------------------------------------------
# Structure handling
# ---------------------------------------------------------------------------


def species_to_smiles(species: dict) -> str:
    """SMILES for a species dict (explicit 'smiles' wins, else adjlist)."""
    smiles = species.get("smiles")
    if smiles:
        return str(smiles)
    adjlist = species.get("adjacency_list")
    if adjlist:
        mol = Molecule.from_adjacency_list(str(adjlist))
        return mol.to_smiles()
    return ""


def reaction_to_smiles(reaction: dict) -> str:
    """Atom-mapped reactants>>products SMILES for a reaction dict.

    'reaction_smiles' is authoritative. Otherwise built unmapped from the
    reactant/product species ('>>' separator; the RIGR featurizer requires
    atom-mapping for full fidelity - the recipe engine, job-05, is where the
    atom correspondence is produced).
    """
    explicit = reaction.get("reaction_smiles")
    if explicit:
        return str(explicit)
    reactants = []
    for sp in reaction.get("reactants", []):
        s = species_to_smiles(sp) if isinstance(sp, dict) else str(sp)
        if s:
            reactants.append(s)
    products = []
    for sp in reaction.get("products", []):
        s = species_to_smiles(sp) if isinstance(sp, dict) else str(sp)
        if s:
            products.append(s)
    if not reactants or not products:
        return ""
    return ">>".join([".".join(reactants), ".".join(products)])


# ---------------------------------------------------------------------------
# The two resolvers
# ---------------------------------------------------------------------------


def estimate_thermo(
    species: dict,
    thermo_db,
    ml,
    counts: Optional[EstimationCounts] = None,
    libraries: Optional[list[str]] = None,
) -> ThermoPrediction:
    """Resolve thermo for a species: library hit -> library value; else ML;
    else MLCoverageError. No other branch exists.

    Args:
        species: dict with at least 'label'; structure as 'smiles' or
            'adjacency_list' (needed for the ML branch).
        thermo_db: a ThermoDB facade (or mock) with
            get_entry_grouped_by_label(label, library=None).
        ml: object with a ``thermo`` attribute (the ThermoML estimator, or
            None when no ML thermo is configured).
        counts: shared EstimationCounts for instrumentation (one is created
            when omitted).
        libraries: optional list of thermo library names to scope the lookup
            (the YAML ``database:`` block); None = all libraries.

    Returns:
        ThermoPrediction (SI: Hf298 J/mol, S298 J/(mol*K), Cp_model with
        .get_heat_capacity(T)).

    Raises:
        MLCoverageError: when neither the libraries nor the ML model covers
            the species. Counts are updated in every outcome.
    """
    if counts is None:
        counts = EstimationCounts()

    label = str(species.get("label", ""))
    entry = _library_entry(thermo_db, label, libraries)
    if entry is not None:
        try:
            H298, S298 = _library_H298_S298(entry)
            cp_model = _library_cp_model(entry)
        except MLCoverageError:
            # The row exists but carries no usable model: a library hit with
            # no value is a coverage gap, not a hit.
            counts.coverage_errors += 1
            raise
        counts.library_hits += 1
        return ThermoPrediction(
            Hf298=H298,
            S298=S298,
            Cp_model=cp_model,
            uncertainties={
                "source": "library",
                "label": label,
                "H298_unit": entry.H298_unit,
                "S298_unit": entry.S298_unit,
            },
        )

    # No library hit -> ML (the ONLY other branch).
    ml_thermo = getattr(ml, "thermo", None)
    smiles = species_to_smiles(species)
    if ml_thermo is None or not smiles:
        counts.coverage_errors += 1
        reason = "no ML thermo estimator configured" if ml_thermo is None else (
            f"no structure (SMILES/adjacency list) for species {label!r}"
        )
        raise MLCoverageError(f"species {label!r}: {reason}")
    try:
        covered = ml_thermo.covers(smiles)
    except Exception as e:  # a malformed structure is a coverage gap
        counts.coverage_errors += 1
        raise MLCoverageError(
            f"species {label!r}: structure check failed ({e})"
        ) from e
    if not covered:
        counts.coverage_errors += 1
        raise MLCoverageError(
            f"species {label!r}: not covered by the ML thermo model "
            f"(SMILES {smiles!r})"
        )
    try:
        prediction = ml_thermo.predict(smiles)
    except MLCoverageError:
        counts.coverage_errors += 1
        raise
    counts.ml_hits += 1
    return prediction


def _library_entry(thermo_db, label: str, libraries: Optional[list[str]]) -> Optional[ThermoEntry]:
    """Resolve the library entry for a label, honouring the run's library list."""
    if not label:
        return None
    if libraries:
        for lib in libraries:
            entry = thermo_db.get_entry_grouped_by_label(label, lib)
            if entry is not None:
                return entry
        return None
    return thermo_db.get_entry_grouped_by_label(label)


def estimate_kinetics(
    reaction: dict,
    kinetics_db,
    ml,
    counts: Optional[EstimationCounts] = None,
    libraries: Optional[list[str]] = None,
    degeneracy: float = 1.0,
) -> tuple[KineticsModel, float]:
    """Resolve kinetics for a reaction: library hit -> library rate model;
    else ML; else MLCoverageError. No other branch exists.

    The library lookup is delegated to the kinetics DB facade (substructure /
    label match over the libraries in scope). A matched reaction whose stored
    rate cannot be assembled (documented rmgdb gap: Chebyshev coefficients,
    or duplicate/seed reactions with no stored kinetics) is NOT a hit - it
    falls through to the ML branch, which is the documented route for exactly
    these (job-02 gap list).

    Args:
        reaction: dict with 'reactants'/'products' (lists of species dicts or
            labels), optional 'label' and optional 'reaction_smiles'
            (atom-mapped reactants>>products; authoritative for the ML
            branch).
        kinetics_db: KineticsDB facade (get_reaction_by_reaction /
            get_rate_model).
        ml: object with a ``kinetics`` attribute (the KineticsML estimator,
            or None).
        counts: shared EstimationCounts.
        libraries: optional list of kinetics library names (scoped lookup).
        degeneracy: the reaction-path degeneracy for the PLAN.md 3b boundary
            conversion of the ML branch: A = 10^pred * degeneracy. The
            reaction object supplies it once job-05/06 wire it (rmgdb stores
            the full-reaction A, so library hits ignore it and return
            degeneracy 1.0).

    Returns:
        (rate_model, degeneracy): the rate model and the degeneracy applied
        (1.0 for library hits - rmgdb stores the full-reaction A; the
        reaction object's own degeneracy is job-05/06 territory - the
        ``degeneracy`` argument for ML hits, since A already carries it).

    Raises:
        MLCoverageError: when neither the libraries nor the ML model covers
            the reaction.
    """
    if counts is None:
        counts = EstimationCounts()

    match = kinetics_db.get_reaction_by_reaction(reaction)
    if match is not None:
        rate = kinetics_db.get_rate_model(match["id"])
        if rate is not None:
            counts.library_hits += 1
            return rate, 1.0
        # Stored reaction whose rate is not assembled (gap) -> ML branch.

    ml_kinetics = getattr(ml, "kinetics", None)
    rxn_smiles = reaction_to_smiles(reaction)
    label = str(reaction.get("label", ""))
    if ml_kinetics is None or not rxn_smiles:
        counts.coverage_errors += 1
        reason = "no ML kinetics estimator configured" if ml_kinetics is None else (
            f"could not build a reaction SMILES for reaction {label!r}"
        )
        raise MLCoverageError(f"reaction {label!r}: {reason}")
    try:
        covered = ml_kinetics.covers(rxn_smiles)
    except Exception as e:
        counts.coverage_errors += 1
        raise MLCoverageError(
            f"reaction {label!r}: structure check failed ({e})"
        ) from e
    if not covered:
        counts.coverage_errors += 1
        raise MLCoverageError(
            f"reaction {label!r}: not covered by the ML kinetics model "
            f"(SMILES {rxn_smiles!r})"
        )
    try:
        prediction = ml_kinetics.predict(rxn_smiles, degeneracy=degeneracy)
    except MLCoverageError:
        counts.coverage_errors += 1
        raise
    counts.ml_hits += 1
    rate_model = Arrhenius(
        A=float(prediction.A),  # CGS cm^3/(mol*s), = 10^pred * degeneracy
        n=float(prediction.n),
        Ea=float(prediction.Ea),
        Tmin=None,
        Tmax=None,
        comment=f"ML estimate, source reaction SMILES {rxn_smiles!r}",
    )
    return rate_model, float(degeneracy)
