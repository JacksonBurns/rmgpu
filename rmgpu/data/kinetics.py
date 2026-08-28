"""Kinetics retrieval logic for rmgpu.

Two layers:

1. ``lookup_kinetics`` - the retrieval entry point: library lookup, with the
   ML-estimator fallback path wired through ``estimate_kinetics`` (job-04).
   It is a thin adapter over the resolver (PLAN.md 3: the retrieval logic
   "is this reaction in a library? if not, estimate" stays; the "estimate"
   branch is one call into the ML model, not a cascade).

2. Rate-model assembly from rmgdb storage (``assemble_rate_model``). rmgdb
   stores the *raw* kinetic parameters in the units given by the library
   files (e.g. ``cm^3/(mol*s)``, ``kcal/mol``); RMG's models evaluate in SI
   (``m^3/(mol*s)``, ``J/mol``). ``assemble_rate_model`` converts them exactly
   the way RMG-Py's ``Quantity`` layer does (factor per unit string, CODATA
   constants per rmgpu/kinetics/models.py) and returns a rmgpu rate model.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Optional

from rmgpu.db.loaders import KineticsDB
from rmgpu.kinetics.models import (
    Arrhenius,
    KineticsModel,
    Lindemann,
    PDepKineticsModel,
    ThirdBody,
    Troe,
    R,
    Na,
)

__all__ = [
    "KineticsLookupResult",
    "lookup_kinetics",
    "A_UNIT_TO_SI",
    "E_UNIT_TO_SI",
    "P_UNIT_TO_SI",
    "arrhenius_params",
    "assemble_rate_model",
]

# ---------------------------------------------------------------------------
# Unit conversions (exactly RMG-Py's Quantity layer factors, SI on the rmgpu
# side). CODATA 2018 via rmgpu.kinetics.models (R, Na).
# ---------------------------------------------------------------------------

# Pre-exponential factors -> SI kunits (m, mol, s)
A_UNIT_TO_SI: dict[str, float] = {
    "s^-1": 1.0,
    "1/s": 1.0,
    "m^3/(mol*s)": 1.0,
    "cm^3/(mol*s)": 1e-6,
    "cm^6/(mol^2*s)": 1e-12,
    "m^6/(mol^2*s)": 1.0,
    # RMG-Py defines molecule = mol / Na, so a per-molecule rate is larger
    # (in mol-based SI) than its per-mol value by a factor of Na:
    # m^3/(molecule*s) -> m^3/(mol*s) multiplies by Na (NOT divides).
    "m^3/(molecule*s)": Na,
    "cm^3/(molecule*s)": 1e-6 * Na,
    "cm^6/(molecule^2*s)": 1e-12 * Na**2,
    "m^2/(mol*s)": 1.0,
    "m^2/(molecule*s)": Na,
}
# tolerate stray parentheses, e.g. "(cm^3/(mol*s))"

# Activation energies -> J/mol
E_UNIT_TO_SI: dict[str, float] = {
    "J/mol": 1.0,
    "kJ/mol": 1e3,
    "cal/mol": 4.184,
    "kcal/mol": 4184.0,
    "eV/molecule": 96485.3423627019,
}

# Pressures -> Pa
P_UNIT_TO_SI: dict[str, float] = {
    "Pa": 1.0,
    "bar": 1e5,
    "atm": 101325.0,
    "torr": 101325.0 / 760.0,
    "psi": 6894.757293168,
}

# Gas concentration unit (for ThirdBody low-pressure A): m^3/(mol*s) base


def _clean_unit(u: Optional[str]) -> Optional[str]:
    if u is None:
        return None
    u = str(u).strip()
    if u.startswith("(") and u.endswith(")"):
        u = u[1:-1]
    return u or None


def convert_A(value: float, unit: Optional[str]) -> float:
    u = _clean_unit(unit)
    if u is None:
        return float(value)
    if u not in A_UNIT_TO_SI:
        raise ValueError(f"Unknown A unit: {unit!r}")
    return float(value) * A_UNIT_TO_SI[u]


def convert_Ea(value: float, unit: Optional[str]) -> float:
    u = _clean_unit(unit)
    if u is None:
        return float(value)
    if u not in E_UNIT_TO_SI:
        raise ValueError(f"Unknown Ea unit: {unit!r}")
    return float(value) * E_UNIT_TO_SI[u]


def convert_P(value: float, unit: Optional[str]) -> float:
    u = _clean_unit(unit)
    if u is None:
        return float(value)
    if u not in P_UNIT_TO_SI:
        raise ValueError(f"Unknown pressure unit: {unit!r}")
    return float(value) * P_UNIT_TO_SI[u]


def _f(x: Any) -> Optional[float]:
    return None if x is None else float(x)


def _t_bounds(row: dict) -> tuple[Optional[float], Optional[float]]:
    return _f(row.get("Tmin_val")), _f(row.get("Tmax_val"))


def arrhenius_params(row: dict) -> dict:
    """Convert one stored Arrhenius parameter row (any *_val columns) to SI.

    Works for kinetics_arrhenius_table, the high/low rows of
    kinetics_lindemann_table / kinetics_troe_table / kinetics_third_body_table,
    and the pressure rows of kinetics_pdep_arrhenius_pressures_table.
    """
    return {
        "A": convert_A(_f(row["A_val"]), row.get("A_unit")),
        "n": _f(row.get("n")) or 0.0,
        "Ea": convert_Ea(_f(row["Ea_val"]), row.get("Ea_unit")),
        "T0": _f(row.get("T0_val")) or 1.0,
        "Tmin": _f(row.get("Tmin_val")),
        "Tmax": _f(row.get("Tmax_val")),
        "comment": row.get("comment") or "",
    }


def _make_arrhenius(p: dict) -> Arrhenius:
    return Arrhenius(
        A=p["A"], n=p["n"], Ea=p["Ea"], T0=p["T0"],
        Tmin=p["Tmin"], Tmax=p["Tmax"], comment=p["comment"],
    )


# ---------------------------------------------------------------------------
# Lookup interface (consumed by job-04/06)
# ---------------------------------------------------------------------------


@dataclass
class KineticsLookupResult:
    """Result of a kinetics lookup."""

    found: bool
    source: str = ""  # "library" or "ml"
    reaction: Optional[dict] = None
    rate_model: Optional[KineticsModel] = None
    degeneracy: float = 1.0  # degeneracy applied to the ML A (library: 1.0)


def lookup_kinetics(
    reaction: dict,
    kinetics_db: KineticsDB,
    ml=None,
    counts=None,
    libraries: Optional[list[str]] = None,
    degeneracy: float = 1.0,
) -> KineticsLookupResult:
    """Look up kinetics for a reaction: library first, else the ML estimator.

    Thin adapter over ``rmgpu.data.estimation.estimate_kinetics`` (the ONLY
    estimation code, job-04). ``ml`` may be None (libraries only - a miss
    then reports ``found=False`` instead of raising); ``counts`` threads the
    instrumentation through so a caller can see the split.
    """
    from rmgpu.data.estimation import estimate_kinetics, MLCoverageError

    try:
        rate_model, deg = estimate_kinetics(
            reaction,
            kinetics_db,
            ml,
            counts=counts,
            libraries=libraries,
            degeneracy=degeneracy,
        )
    except MLCoverageError:
        return KineticsLookupResult(found=False, source="")
    # Label the branch that won (the counts object carries the split when
    # the caller instruments; this tag just names the source).
    match = kinetics_db.get_reaction_by_reaction(reaction)
    source = (
        "library"
        if match is not None and kinetics_db.get_rate_model(match["id"]) is not None
        else "ml"
    )
    return KineticsLookupResult(
        found=True, source=source, reaction=match, rate_model=rate_model,
        degeneracy=deg,
    )


# ---------------------------------------------------------------------------
# Rate-model assembly from rmgdb tables
# ---------------------------------------------------------------------------


def _table_row(cur, table: str, reaction_id: int) -> Optional[dict]:
    cur.execute(f"SELECT * FROM {table} WHERE library_reaction_id = ? LIMIT 1", (reaction_id,))
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    return dict(zip(cols, row)) if row else None


def _table_rows(cur, table: str, reaction_id: int) -> list[dict]:
    cur.execute(f"SELECT * FROM {table} WHERE library_reaction_id = ?", (reaction_id,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _rows_where(cur, table: str, column: str, value) -> list[dict]:
    cur.execute(f"SELECT * FROM {table} WHERE {column} = ?", (value,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _build_multi_arrhenius(rows: list[dict]):
    """MultiArrhenius: sum of temperature-banded Arrhenius models, ordered by Tmin."""
    rows = sorted(rows, key=lambda r: (_f(r.get("Tmin_val")) or 0.0, _f(r.get("Tmax_val")) or 0.0))
    arrhs = [_make_arrhenius(arrhenius_params(r)) for r in rows]
    from rmgpu.kinetics.models import KineticsModel

    class _MultiArrhenius(KineticsModel):
        """Sum of temperature-banded Arrhenius models (RMG MultiArrhenius parity)."""

        def __init__(self, arrhenius, Tmin=None, Tmax=None):
            super().__init__(Tmin=Tmin, Tmax=Tmax)
            self.arrhenius = arrhenius

        def get_rate_coefficient(self, T, P=0.0, **kw):
            return sum(a.get_rate_coefficient(T) for a in self.arrhenius)

    Tmin = min(a.Tmin for a in arrhs if a.Tmin is not None) if any(a.Tmin for a in arrhs) else None
    Tmax = max(a.Tmax for a in arrhs if a.Tmax is not None) if any(a.Tmax for a in arrhs) else None
    return _MultiArrhenius(arrhs, Tmin=Tmin, Tmax=Tmax)


def _build_pdep_arrhenius(pdep_row: dict, pressure_rows: list[dict]):
    """PDepArrhenius: log-log pressure interpolation between stored pressure
    points, exactly RMG-Py's get_adjacent_expressions + rate formula."""
    import math

    pts = sorted(
        (
            (
                convert_P(_f(r["P_val"]), r.get("P_unit")),
                _make_arrhenius(
                    {
                        "A": convert_A(_f(r["A_val"]), r.get("A_unit")),
                        "n": _f(r.get("n")) or 0.0,
                        "Ea": convert_Ea(_f(r["Ea_val"]), r.get("Ea_unit")),
                        "T0": 1.0,  # stored PDepArrhenius sub-models use T0=1K
                        "Tmin": _f(r.get("Tmin_val")),
                        "Tmax": _f(r.get("Tmax_val")),
                        "comment": r.get("comment") or "",
                    }
                ),
            )
            for r in pressure_rows
        ),
        key=lambda t: t[0],
    )
    pressures = [p for p, _ in pts]
    arrhs = [a for _, a in pts]
    Tmin = _f(pdep_row.get("Tmin_val"))
    Tmax = _f(pdep_row.get("Tmax_val"))

    class _PDepArrhenius(PDepKineticsModel):
        """RMG-Py PDepArrhenius evaluation: log-log interpolation in P."""

        def __init__(self, pressures, arrhenius, Tmin=None, Tmax=None):
            super().__init__(Tmin=Tmin, Tmax=Tmax, Pmin=pressures[0], Pmax=pressures[-1])
            self.pressures = pressures
            self.arrhenius = arrhenius

        def get_adjacent_expressions(self, P):
            ilow = 0
            ihigh = -1
            for i, p in enumerate(self.pressures):
                if p <= P:
                    ilow = i
                if p >= P and ihigh == -1:
                    ihigh = i
            if ihigh == -1:
                ihigh = len(self.pressures) - 1
            return (
                self.pressures[ilow],
                self.pressures[ihigh],
                self.arrhenius[ilow],
                self.arrhenius[ihigh],
            )

        def get_rate_coefficient(self, T, P=0.0, **kw):
            if P == 0:
                raise ValueError("No pressure specified to PDepArrhenius.")
            Plow, Phigh, alow, ahigh = self.get_adjacent_expressions(P)
            if Plow == Phigh:
                return alow.get_rate_coefficient(T)
            klow = alow.get_rate_coefficient(T)
            khigh = ahigh.get_rate_coefficient(T)
            if klow == khigh == 0.0:
                return 0.0
            return klow * 10 ** (math.log10(P / Plow) / math.log10(Phigh / Plow) * math.log10(khigh / klow))

    return _PDepArrhenius(pressures, arrhs, Tmin=Tmin, Tmax=Tmax)


def _build_multi_pdep_arrhenius(cur, pdep_rows: list[dict]):
    """MultiPDepArrhenius: sum of temperature-banded PDepArrhenius blocks.

    rmgdb stores one kinetics_pdep_arrhenius row per PDepArrhenius block;
    RMG-Py's MultiPDepArrhenius sums k(T) over its ``arrhenius`` list, so the
    rmgpu model does the same.
    """
    from rmgpu.kinetics.models import KineticsModel

    blocks = []
    for row in pdep_rows:
        ppts = _rows_where(cur, "kinetics_pdep_arrhenius_pressures_table",
                           "pdep_id", int(row["id"]))
        blocks.append(_build_pdep_arrhenius(row, ppts))

    class _MultiPDepArrhenius(KineticsModel):
        """Sum of temperature-banded PDepArrhenius blocks (RMG parity)."""

        def __init__(self, arrhenius, Tmin=None, Tmax=None):
            super().__init__(Tmin=Tmin, Tmax=Tmax)
            self.arrhenius = arrhenius

        def get_rate_coefficient(self, T, P=0.0, **kw):
            return sum(b.get_rate_coefficient(T, P) for b in self.arrhenius)

    Tmin = min((b.Tmin for b in blocks if b.Tmin is not None), default=None)
    Tmax = max((b.Tmax for b in blocks if b.Tmax is not None), default=None)
    return _MultiPDepArrhenius(blocks, Tmin=Tmin, Tmax=Tmax)


def assemble_rate_model(db_path: str, reaction_id: int) -> Optional[KineticsModel]:
    """Build the rmgpu rate model for a kinetics library reaction (SI units).

    Reads the rate tables for ``reaction_id`` and assembles:
      - third_body row    -> ThirdBody (low-limit Arrhenius from its table)
      - 1 Arrhenius row   -> Arrhenius
      - n Arrhenius rows  -> MultiArrhenius (temperature banded)
      - lindemann row     -> Lindemann
      - troe row          -> Troe
      - chebyshev row     -> None (COEFFICIENTS NOT STORED by rmgdb - gap)
      - pdep_arrhenius    -> PDepArrhenius
    Returns None if the reaction has no stored rate data (duplicate/seed
    reactions carry no kinetics).
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        arr_rows = _table_rows(cur, "kinetics_arrhenius_table", reaction_id)
        third_body_row = _table_row(cur, "kinetics_third_body_table", reaction_id)
        lind_row = _table_row(cur, "kinetics_lindemann_table", reaction_id)
        troe_row = _table_row(cur, "kinetics_troe_table", reaction_id)
        cheb_row = _table_row(cur, "kinetics_chebyshev_table", reaction_id)
        pdep_row = _table_row(cur, "kinetics_pdep_arrhenius_table", reaction_id)
        if cheb_row is not None:
            return None  # coefficients table empty in rmgdb (documented gap)
        if third_body_row is not None:
            low = _make_arrhenius(arrhenius_params({
                "A_val": third_body_row["low_A_val"], "A_unit": third_body_row["low_A_unit"],
                "n": third_body_row["low_n"], "Ea_val": third_body_row["low_Ea_val"],
                "Ea_unit": third_body_row["low_Ea_unit"], "T0_val": 1.0,
                "Tmin_val": third_body_row["Tmin_val"], "Tmin_unit": third_body_row["Tmin_unit"],
                "Tmax_val": third_body_row["Tmax_val"], "Tmax_unit": third_body_row["Tmax_unit"],
                "comment": third_body_row.get("comment"),
            }))
            return ThirdBody(arrheniusLow=low,
                             Tmin=_f(third_body_row["Tmin_val"]),
                             Tmax=_f(third_body_row["Tmax_val"]))
        if troe_row is not None:
            high = _make_arrhenius(arrhenius_params({
                "A_val": troe_row["high_A_val"], "A_unit": troe_row["high_A_unit"],
                "n": troe_row["high_n"], "Ea_val": troe_row["high_Ea_val"],
                "Ea_unit": troe_row["high_Ea_unit"], "T0_val": 1.0,
                "Tmin_val": troe_row["Tmin_val"], "Tmin_unit": troe_row["Tmin_unit"],
                "Tmax_val": troe_row["Tmax_val"], "Tmax_unit": troe_row["Tmax_unit"],
                "comment": troe_row.get("comment"),
            }))
            low = _make_arrhenius(arrhenius_params({
                "A_val": troe_row["low_A_val"], "A_unit": troe_row["low_A_unit"],
                "n": troe_row["low_n"], "Ea_val": troe_row["low_Ea_val"],
                "Ea_unit": troe_row["low_Ea_unit"], "T0_val": 1.0,
                "Tmin_val": troe_row["Tmin_val"], "Tmin_unit": troe_row["Tmin_unit"],
                "Tmax_val": troe_row["Tmax_val"], "Tmax_unit": troe_row["Tmax_unit"],
                "comment": troe_row.get("comment"),
            }))
            return Troe(arrheniusHigh=high, arrheniusLow=low,
                        alpha=_f(troe_row["alpha"]) or 0.0,
                        T3=_f(troe_row["T3_val"]) or 0.0,
                        T1=_f(troe_row["T1_val"]) or 0.0,
                        T2=_f(troe_row["T2_val"]) or 0.0,
                        Tmin=_f(troe_row["Tmin_val"]), Tmax=_f(troe_row["Tmax_val"]))
        if lind_row is not None:
            high = _make_arrhenius(arrhenius_params({
                "A_val": lind_row["high_A_val"], "A_unit": lind_row["high_A_unit"],
                "n": lind_row["high_n"], "Ea_val": lind_row["high_Ea_val"],
                "Ea_unit": lind_row["high_Ea_unit"], "T0_val": 1.0,
                "Tmin_val": lind_row["Tmin_val"], "Tmin_unit": lind_row["Tmin_unit"],
                "Tmax_val": lind_row["Tmax_val"], "Tmax_unit": lind_row["Tmax_unit"],
                "comment": lind_row.get("comment"),
            }))
            low = _make_arrhenius(arrhenius_params({
                "A_val": lind_row["low_A_val"], "A_unit": lind_row["low_A_unit"],
                "n": lind_row["low_n"], "Ea_val": lind_row["low_Ea_val"],
                "Ea_unit": lind_row["low_Ea_unit"], "T0_val": 1.0,
                "Tmin_val": lind_row["Tmin_val"], "Tmin_unit": lind_row["Tmin_unit"],
                "Tmax_val": lind_row["Tmax_val"], "Tmax_unit": lind_row["Tmax_unit"],
                "comment": lind_row.get("comment"),
            }))
            return Lindemann(arrheniusHigh=high, arrheniusLow=low,
                             Tmin=_f(lind_row["Tmin_val"]), Tmax=_f(lind_row["Tmax_val"]))
        if pdep_row is not None:
            # A reaction may carry ONE PDepArrhenius block or several
            # (MultiPDepArrhenius) - rmgdb stores one kinetics_pdep_arrhenius
            # row per PDepArrhenius block, all sharing library_reaction_id.
            # pressures link via pdep_id -> kinetics_pdep_arrhenius_table.id.
            pdep_rows = _table_rows(cur, "kinetics_pdep_arrhenius_table", reaction_id)
            if len(pdep_rows) > 1:
                return _build_multi_pdep_arrhenius(cur, pdep_rows)
            ppts = _rows_where(cur, "kinetics_pdep_arrhenius_pressures_table",
                               "pdep_id", int(pdep_row["id"]))
            return _build_pdep_arrhenius(pdep_row, ppts)
        if len(arr_rows) > 1:
            return _build_multi_arrhenius(arr_rows)
        if len(arr_rows) == 1:
            row = arr_rows[0]
            p = arrhenius_params(row)
            # RMG loads a single Arrhenius as ThirdBody when it is a 3rd-body
            # (units m^3/(mol*s) with the reaction being association); the
            # table does not distinguish, so we expose it as Arrhenius here -
            # consumers needing ThirdBody semantics wrap it (job-06).
            return _make_arrhenius(p)
    return None
