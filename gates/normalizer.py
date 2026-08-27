"""Shared normalization for the job-02 gate (content hash + lookup parity).

This module is the SINGLE source of the canonical "entry dict" format used by
BOTH sides of the gate:

  * the rmgdb / rmgpu side (reads the flat SQL views, or RMG-Py objects built
    from the same dict), and
  * the RMG-Py side (loads the raw library objects, then normalizes them with
    the same functions).

It is deliberately pure Python (stdlib + yaml only) so it can be imported
unchanged from both the ``rmgpu`` and ``rmg_env`` conda environments.

Why a normalizer at all
-----------------------
The rmgdb thermo views are flat LEFT-JOINs, so one logical library entry fans
out to several rows:
  * a NASA7 entry with *n* polynomials yields *n* rows (c1 set, Tdata NULL);
  * a ThermoData entry yields one row (Tdata set, c1 NULL);
  * 16 labels in primaryThermoLibrary are stored as TWO duplicate rows in
    ``thermo_libraries_table`` (a rmgdb build artifact), which multiplies the
    fan-out again.
The normalizer groups the view rows by label and deduplicates them back into
one canonical entry, exactly mirroring how RMG-Py holds a single entry per
label. The entry *count* the gate compares uses ``COUNT(DISTINCT label)``,
which already accounts for the duplicate rows.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Unit-conversion factors. These EXACTLY reproduce RMG-Py's Quantity layer
# (rmgpy.quantity.Quantity, backed by the `quantities` package with
# molecule = mol / 6.02214179e23). They must match to double precision for the
# 1e-10 rate round-trip to hold.
# ---------------------------------------------------------------------------
_NA = 6.02214179e23  # RMG-Py Avogadro (rmgpy.constants.Na)

# Pre-exponential factor -> SI m^3/(mol*s)
A_UNIT_FACTORS = {
    "s^-1": 1.0,
    "1/s": 1.0,
    "m^3/(mol*s)": 1.0,
    "cm^3/(mol*s)": 1e-6,
    "cm^6/(mol^2*s)": 1e-12,
    "m^6/(mol^2*s)": 1.0,
    "m^3/(molecule*s)": _NA,
    "cm^3/(molecule*s)": 1e-6 * _NA,
    "cm^6/(molecule^2*s)": 1e-12 * _NA ** 2,
    "m^2/(mol*s)": 1.0,
    "m^2/(molecule*s)": _NA,
}

# Activation energy -> J/mol
E_UNIT_FACTORS = {
    "J/mol": 1.0,
    "kJ/mol": 1000.0,
    "cal/mol": 4.184,
    "kcal/mol": 4184.0,
    "eV/molecule": 96485.3423627019,
}

# Pressure -> Pa
P_UNIT_FACTORS = {
    "Pa": 1.0,
    "bar": 1e5,
    "atm": 101325.0,
    "torr": 101325.0 / 760.0,
    "psi": 6894.757293168359,
}


def clean_unit(u: Optional[str]) -> Optional[str]:
    """Strip whitespace and any surrounding parentheses from a unit string."""
    if u is None:
        return None
    u = str(u).strip()
    if u.startswith("(") and u.endswith(")"):
        u = u[1:-1]
    return u or None


def convert_A(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    u = clean_unit(unit)
    if u is None:
        return float(value)
    return float(value) * A_UNIT_FACTORS[u]


def convert_Ea(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    u = clean_unit(unit)
    if u is None:
        return float(value)
    return float(value) * E_UNIT_FACTORS[u]


def convert_P(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    u = clean_unit(unit)
    if u is None:
        return float(value)
    return float(value) * P_UNIT_FACTORS[u]


# ---------------------------------------------------------------------------
# Canonical hashing
# ---------------------------------------------------------------------------
# rmgdb stores the correctly-rounded double of a file literal (e.g. -0.0010244)
# while RMG-Py's parse of the same literal can hold a 1-3 ulp-different double
# (-0.0010243999999999997). That is parse noise, not data corruption. The gate
# requires "byte-identical after normalization", so the normalization absorbs
# this by rounding every float to 9 significant figures before hashing. 9 sig
# figs is far below any real corruption (which shifts a coefficient by >>1e-9
# relative) yet far above double-ulp parse noise. Full precision is kept in the
# returned dicts so the rate round-trip (rel tol 1e-10) is unaffected.

import yaml  # noqa: E402  (pure-python; present in both gate envs)


def _round_sig(x: float, sig: int = 9) -> float:
    if x == 0.0 or not math.isfinite(x):
        return x
    return round(x, sig - 1 - int(math.floor(math.log10(abs(x)))))


def _round_floats(obj):
    if isinstance(obj, float):
        return _round_sig(obj)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v) for v in obj]
    return obj


def canonical_dump(entry: Any) -> str:
    """Deterministic YAML text of ``entry`` with floats rounded to 9 sig figs.

    Both sides of the gate feed their (full-precision) entry dict through this
    to produce the byte-comparable string whose SHA-256 is the content hash.
    """
    return yaml.safe_dump(
        _round_floats(entry), sort_keys=True, default_flow_style=False,
    )


# ---------------------------------------------------------------------------
# Thermo normalization
# ---------------------------------------------------------------------------

def _is_null(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _f(v: Any) -> Optional[float]:
    return None if _is_null(v) else float(v)


def normalize_thermo_entry(label: str, rows: list[dict]) -> dict:
    """Collapse the (possibly fanned-out) view rows for one label into one
    canonical thermo entry dict.

    ``rows`` are dicts with the ``thermo_libraries_view`` columns. The result
    is identical whether it was built from the SQL view or from a RMG-Py entry
    (see ``_rows_from_rmgpy_thermo``).
    """
    td = None
    nasa_segs: dict[tuple, dict] = {}
    nasa_Tmin = nasa_Tmax = E0 = E0_unit = None

    for row in rows:
        # scalar fields: first non-NULL wins (duplicates carry identical data)
        if td is None and not _is_null(row.get("Tdata_1")):
            Tdata, Cpdata = [], []
            for i in range(1, 8):
                t = row.get(f"Tdata_{i}")
                c = row.get(f"Cpdata_{i}")
                if not _is_null(t):
                    Tdata.append(float(t))
                    if not _is_null(c):
                        Cpdata.append(float(c))
            td = {
                "H298": _f(row.get("H298")),
                "H298_unit": row.get("H298_unit"),
                "S298": _f(row.get("S298")),
                "S298_unit": row.get("S298_unit"),
                "Tdata": Tdata,
                "Cpdata": Cpdata,
                "Tdata_unit": row.get("Tdata_unit"),
                "Cpdata_unit": row.get("Cpdata_unit"),
            }
        if not _is_null(row.get("c1")):
            seg = {
                "c1": _f(row.get("c1")), "c2": _f(row.get("c2")),
                "c3": _f(row.get("c3")), "c4": _f(row.get("c4")),
                "c5": _f(row.get("c5")), "c6": _f(row.get("c6")),
                "c7": _f(row.get("c7")),
                "Tmin": _f(row.get("poly_Tmin")),
                "Tmax": _f(row.get("poly_Tmax")),
            }
            nasa_segs[(seg["c1"], seg["Tmin"])] = seg
            if nasa_Tmin is None or not _is_null(row.get("nasa_Tmin")):
                nasa_Tmin = _f(row.get("nasa_Tmin"))
                nasa_Tmax = _f(row.get("nasa_Tmax"))
            if not _is_null(row.get("E0")):
                E0 = _f(row.get("E0"))
                E0_unit = row.get("E0_unit")

    # NOTE: rmgdb does NOT store the Wilhoit fit coefficients (a0..a3, B, H0,
    # S0) or Cp0/CpInf - those view columns are entirely NULL (verified). A
    # ThermoData entry is therefore emitted as the raw data points, matching
    # what RMG-Py keeps verbatim. Documented rmgdb gap.
    # (short/long descriptions are prose, not data - intentionally excluded.)
    entry: dict[str, Any] = {
        "label": label,
        "model": "ThermoData" if td is not None else ("NASA" if nasa_segs else "none"),
    }
    if td is not None:
        entry["Tdata"] = td["Tdata"]
        entry["Tdata_unit"] = td["Tdata_unit"]
        entry["Cpdata"] = td["Cpdata"]
        entry["Cpdata_unit"] = td["Cpdata_unit"]
        entry["H298"] = td["H298"]
        entry["H298_unit"] = td["H298_unit"]
        entry["S298"] = td["S298"]
        entry["S298_unit"] = td["S298_unit"]
    if nasa_segs:
        segs = sorted(nasa_segs.values(), key=lambda s: (s["Tmin"] or 0.0, s["Tmax"] or 0.0))
        entry["nasa"] = segs
        entry["nasa_Tmin"] = nasa_Tmin
        entry["nasa_Tmax"] = nasa_Tmax
        entry["E0"] = E0
        entry["E0_unit"] = E0_unit
    return entry


def _val(q) -> Optional[float]:
    """Scalar value of a RMG-Py Quantity (or plain number) -> float, else None."""
    if q is None:
        return None
    if hasattr(q, "value"):
        v = q.value
        return None if v is None else float(v)
    try:
        return float(q)
    except (TypeError, ValueError):
        return None


def _arr_of(q) -> list[float]:
    """List of scalar values of a RMG-Py ArrayQuantity (or list), else []"""
    if q is None:
        return []
    if hasattr(q, "value"):
        q = q.value
    if q is None:
        return []
    return [float(x) for x in list(q)]


def _rows_from_rmgpy_thermo(entry) -> list[dict]:
    """Build view-shaped row dicts from a RMG-Py thermo Entry, so the SAME
    ``normalize_thermo_entry`` can be used on both sides."""
    data = entry.data
    rows = []
    model = type(data).__name__
    if model == "ThermoData":
        Tdata = _arr_of(data.Tdata)
        Cpdata = _arr_of(data.Cpdata)
        row: dict[str, Any] = {
            "H298": _val(data.H298),
            "H298_unit": getattr(data.H298, "units", None),
            "S298": _val(data.S298),
            "S298_unit": getattr(data.S298, "units", None),
            "Tdata_unit": getattr(data.Tdata, "units", None),
            "Cpdata_unit": getattr(data.Cpdata, "units", None),
        }
        for i in range(7):
            row[f"Tdata_{i+1}"] = Tdata[i] if i < len(Tdata) else None
            row[f"Cpdata_{i+1}"] = Cpdata[i] if i < len(Cpdata) else None
        rows.append(row)
    elif model == "NASA":
        nasa = {
            "nasa_Tmin": _val(data.Tmin),
            "nasa_Tmax": _val(data.Tmax),
            "E0": _val(getattr(data, "E0", None)),
            "E0_unit": getattr(getattr(data, "E0", None), "units", None),
        }
        for poly in data.polynomials:
            row = dict(nasa)
            coeffs = _arr_of(poly.coeffs)
            for i in range(7):
                row[f"c{i+1}"] = coeffs[i] if i < len(coeffs) else None
            row["poly_Tmin"] = _val(poly.Tmin)
            row["poly_Tmax"] = _val(poly.Tmax)
            rows.append(row)
    # Wilhoit / other models: rows stay empty -> model "none" (documented gap)
    return rows


# ---------------------------------------------------------------------------
# Kinetics normalization
# ---------------------------------------------------------------------------

def _arr_from_row(r: dict, akey="A_val", nkey="n", ekey="Ea_val", t0key="T0_val", bounds=True) -> dict:
    """Normalize one stored Arrhenius row (rmgdb *_table) to a canonical dict.

    ``bounds`` controls whether the row's Tmin/Tmax are included. For nested
    high/low Arrhenius inside a falloff model, rmgdb stores only the
    MODEL-level Tmin/Tmax (a single column pair), NOT per-nested bounds - so the
    nested canonical dict omits them (documented rmgdb gap; the nested bounds
    are validity metadata, unused by get_rate_coefficient).
    """
    a_unit = r.get(akey.replace("_val", "_unit"))
    e_unit = r.get(ekey.replace("_val", "_unit"))
    out = {
        "A": convert_A(_f(r.get(akey)), a_unit),
        "n": _f(r.get(nkey)) or 0.0,
        "Ea": convert_Ea(_f(r.get(ekey)), e_unit),
        "T0": _f(r.get(t0key)) or 1.0,
    }
    if bounds:
        out["Tmin"] = _f(r.get("Tmin_val"))
        out["Tmax"] = _f(r.get("Tmax_val"))
    return out


def normalize_kinetics_entry(label: str, row: dict) -> dict:
    """Build the canonical kinetics entry dict from a RAW reaction row.

    ``row`` must carry the keys used by ``KineticsDB.dump_raw_reaction``:
      - "label", and exactly one of the model payloads:
        "arr" (list of arrhenius rows), "thirdbody", "lindemann", "troe",
        "chebyshev", "pdep" (list of pdep header rows, each with "pressures").
    """
    out: dict[str, Any] = {}
    out["label"] = label

    if row.get("troe") is not None:
        t = row["troe"]
        out["model"] = "Troe"
        out["high"] = _arr_from_row(t, "high_A_val", "high_n", "high_Ea_val", bounds=False)
        out["low"] = _arr_from_row(t, "low_A_val", "low_n", "low_Ea_val", bounds=False)
        out["alpha"] = _f(t.get("alpha")) or 0.0
        out["T1"] = _f(t.get("T1_val"))
        out["T2"] = _f(t.get("T2_val"))
        out["T3"] = _f(t.get("T3_val"))
        out["Tmin"] = _f(t.get("Tmin_val"))
        out["Tmax"] = _f(t.get("Tmax_val"))
    elif row.get("lindemann") is not None:
        t = row["lindemann"]
        out["model"] = "Lindemann"
        out["high"] = _arr_from_row(t, "high_A_val", "high_n", "high_Ea_val", bounds=False)
        out["low"] = _arr_from_row(t, "low_A_val", "low_n", "low_Ea_val", bounds=False)
        out["Tmin"] = _f(t.get("Tmin_val"))
        out["Tmax"] = _f(t.get("Tmax_val"))
    elif row.get("thirdbody") is not None:
        t = row["thirdbody"]
        out["model"] = "ThirdBody"
        out["low"] = _arr_from_row(t, "low_A_val", "low_n", "low_Ea_val", bounds=False)
        out["Tmin"] = _f(t.get("Tmin_val"))
        out["Tmax"] = _f(t.get("Tmax_val"))
        effs = row.get("efficiencies") or {}
        out["efficiencies"] = {k: float(v) for k, v in sorted(effs.items())}
    elif row.get("chebyshev") is not None:
        t = row["chebyshev"]
        out["model"] = "Chebyshev"
        # coefficients are NOT stored by rmgdb (documented gap)
        out["Tmin"] = _f(t.get("Tmin_val"))
        out["Tmax"] = _f(t.get("Tmax_val"))
        out["degreeT"] = _f(t.get("degreeT"))
        out["degreeP"] = _f(t.get("degreeP"))
    elif row.get("pdep"):
        pd = row["pdep"]
        if len(pd) > 1:
            out["model"] = "MultiPDepArrhenius"
            out["Tmin"] = _f(pd[0].get("Tmin_val"))
            out["Tmax"] = _f(pd[0].get("Tmax_val"))
            out["pdeps"] = [_pdep_from_header(h) for h in pd]
        else:
            h = pd[0]
            out["model"] = "PDepArrhenius"
            out["Tmin"] = _f(h.get("Tmin_val"))
            out["Tmax"] = _f(h.get("Tmax_val"))
            out["pdep"] = _pdep_from_header(h)
    else:
        arr = row.get("arr") or []
        if not arr:
            out["model"] = "none"
        elif len(arr) == 1:
            out["model"] = "Arrhenius"
            out.update(_arr_from_row(arr[0]))
        else:
            out["model"] = "MultiArrhenius"
            out["arr"] = [_arr_from_row(a) for a in arr]
    return out


def _pdep_from_header(h: dict) -> dict:
    pts = h.get("pressures") or []
    pressures = sorted(convert_P(_f(p.get("P_val")), p.get("P_unit")) for p in pts)
    arr = []
    for p in sorted(pts, key=lambda q: (convert_P(_f(q.get("P_val")), q.get("P_unit")) or 0.0)):
        arr.append({
            "A": convert_A(_f(p.get("A_val")), p.get("A_unit")),
            "n": _f(p.get("n")) or 0.0,
            "Ea": convert_Ea(_f(p.get("Ea_val")), p.get("Ea_unit")),
        })
    return {"pressures": pressures, "arrhenius": arr}


def _raw_from_rmgpy_kinetics(entry) -> dict:
    """Build a raw reaction row from a RMG-Py kinetics Entry (mirrors the
    rmgdb tables), so ``normalize_kinetics_entry`` works on both sides."""
    d = entry.data
    m = type(d).__name__
    row: dict[str, Any] = {"label": entry.label}
    if m == "Arrhenius":
        row["arr"] = [_rmg_arr_to_row(d)]
    elif m == "MultiArrhenius":
        row["arr"] = [_rmg_arr_to_row(a) for a in d.arrhenius]
    elif m == "ThirdBody":
        # rmgdb stores only MODEL-level bounds for falloff models, not the
        # nested arrheniusLow/High bounds -> drop nested bounds (documented gap).
        row["thirdbody"] = _rmg_arr_to_row(d.arrheniusLow, "low", bounds=False)
        row["thirdbody"]["Tmin_val"] = _val(d.Tmin)
        row["thirdbody"]["Tmax_val"] = _val(d.Tmax)
        effs = getattr(d, "efficiencies", None) or {}
        # RMG-Py keys efficiencies by Molecule; rmgdb stores the SMILES label.
        row["efficiencies"] = {
            (k.to_smiles() if hasattr(k, "to_smiles") else str(k)): float(v)
            for k, v in effs.items()
        }
    elif m == "Lindemann":
        row["lindemann"] = {**_rmg_arr_to_row(d.arrheniusHigh, "high", bounds=False),
                            **_rmg_arr_to_row(d.arrheniusLow, "low", bounds=False),
                            "Tmin_val": _val(d.Tmin), "Tmax_val": _val(d.Tmax)}
    elif m == "Troe":
        row["troe"] = {**_rmg_arr_to_row(d.arrheniusHigh, "high", bounds=False),
                       **_rmg_arr_to_row(d.arrheniusLow, "low", bounds=False),
                       "alpha": _val(d.alpha) or 0.0,
                       "T1_val": _val(d.T1), "T2_val": _val(d.T2), "T3_val": _val(d.T3),
                       "Tmin_val": _val(d.Tmin), "Tmax_val": _val(d.Tmax)}
    elif m == "Chebyshev":
        row["chebyshev"] = {"Tmin_val": _val(d.Tmin), "Tmax_val": _val(d.Tmax),
                            "degreeT": _val(getattr(d, "degreeT", None)),
                            "degreeP": _val(getattr(d, "degreeP", None))}
    elif m == "PDepArrhenius":
        row["pdep"] = [_rmg_pdep_header(d)]
    elif m == "MultiPDepArrhenius":
        row["pdep"] = [_rmg_pdep_header(a) for a in d.arrhenius]
    return row


def _rmg_arr_to_row(a, prefix="", bounds=True) -> dict:
    p = prefix + "_" if prefix else ""
    out = {
        f"{p}A_val": _val(a.A),
        f"{p}A_unit": getattr(a.A, "units", None),
        f"{p}n": _val(a.n) or 0.0,
        f"{p}Ea_val": _val(a.Ea),
        f"{p}Ea_unit": getattr(a.Ea, "units", None),
        f"{p}T0_val": _val(a.T0) or 1.0,
    }
    if bounds:
        out["Tmin_val"] = _val(a.Tmin)
        out["Tmax_val"] = _val(a.Tmax)
    return out


def _rmg_pdep_header(d) -> dict:
    pts = []
    punit = getattr(d.pressures, "units", None)
    for p, a in zip(_arr_of(d.pressures), d.arrhenius):
        pts.append({
            "P_val": _val(p), "P_unit": punit,
            "A_val": _val(a.A), "A_unit": getattr(a.A, "units", None),
            "n": _val(a.n) or 0.0,
            "Ea_val": _val(a.Ea), "Ea_unit": getattr(a.Ea, "units", None),
        })
    return {"Tmin_val": _val(d.Tmin), "Tmax_val": _val(d.Tmax), "pressures": pts}
