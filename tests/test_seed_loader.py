"""job-06/step-07: seed-loader honesty tests + gate physical-validity test.

Covers:
  * the c3h4 seed name ('GRI-Mech3.0-N') resolves (alias) to the real rmgdb
    library and loads a NON-EMPTY species list AND a NON-EMPTY reaction list;
  * name resolution is loud: an unknown name RAISES ValueError;
  * missing species are SKIPPED (counted), never turned into a methane
    placeholder (a sqlite-based fixture with a dangling label);
  * termolecular A (cm^6/(mol^2*s)) is converted by the ACTUAL unit (1e-12,
    not the old 1e-6 substring heuristic) and cal/mol Ea is converted;
  * multi-band (2-row) Arrhenius is dropped + counted, not silently LIMIT-1.
  * the gate's physical-validity check FAILs a profile with a mole fraction
    outside [0,1] and PASSes a physical one.
"""
import csv
import os
import sqlite3
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rmgpu.db import Databases  # noqa: E402
from rmgpu.db.seed_loader import (  # noqa: E402
    SEED_LIBRARY_ALIASES,
    _library_id_by_name,
    _normalize_library_name,
    _flat_rate_param,
    load_seed_mechanism,
)
import gates.gate_06 as G  # noqa: E402


REAL_DB = "/home/jackson/rmgpu/rmgdb/db"

# Real, valid rmgdb-style adjacency lists (from the GRI-Mech3 dictionary).
ADJ_H2 = "1 H u0 p0 c0 {2,S}\n2 H u0 p0 c0 {1,S}\n"
ADJ_H = "multiplicity 2\n1 H u1 p0 c0\n"
ADJ_O2 = ("multiplicity 3\n"
          "1 O u1 p2 c0 {2,S}\n"
          "2 O u1 p2 c0 {1,S}\n")
ADJ_C2H4 = ("1 C u0 p0 c0 {2,D} {3,S} {4,S}\n"
            "2 C u0 p0 c0 {1,D} {5,S} {6,S}\n"
            "3 H u0 p0 c0 {1,S}\n"
            "4 H u0 p0 c0 {1,S}\n"
            "5 H u0 p0 c0 {2,S}\n"
            "6 H u0 p0 c0 {2,S}\n")


def _db():
    return Databases.from_config(
        {"paths": {"thermo": os.path.join(REAL_DB, "thermo.db"),
                   "kinetics": os.path.join(REAL_DB, "kinetics.db")},
         "thermo_libraries": ["primaryThermoLibrary", "GRI-Mech3.0-N"],
         "reaction_libraries": []})


# ---------------------------------------------------------------------------
# Fix 1: name resolution
# ---------------------------------------------------------------------------
def test_normalize_library_name():
    assert _normalize_library_name("GRI-Mech3.0-N") == "GRI-Mech3"
    assert _normalize_library_name("GRI-Mech3.0") == "GRI-Mech3"
    assert _normalize_library_name("GRI-Mech3") == "GRI-Mech3"


def test_alias_table_has_c3h4_seed_name():
    assert SEED_LIBRARY_ALIASES.get("GRI-Mech3.0-N") == "GRI-Mech3"


def test_library_id_by_name_exact_and_alias_and_miss():
    conn = sqlite3.connect(os.path.join(REAL_DB, "kinetics.db"))
    try:
        lid, name, how = _library_id_by_name(conn, "GRI-Mech3")
        assert lid is not None and name == "GRI-Mech3" and how == "exact"
        lid, name, how = _library_id_by_name(conn, "GRI-Mech3.0-N")
        assert lid is not None and name == "GRI-Mech3" and how.startswith("alias")
        assert _library_id_by_name(conn, "No-Such-Mech-9.9") == (None, None, None)
    finally:
        conn.close()


def test_unknown_seed_name_raises():
    """Loud resolution (Fix 1): an unresolvable name raises ValueError."""
    with pytest.raises(ValueError):
        load_seed_mechanism("No-Such-Mech-9.9", _db())


def test_c3h4_seed_name_loads_nonempty_species_and_reactions():
    """The c3h4.yaml seed name must load a real mechanism (the bug B1 fix):
    non-empty species AND non-empty reactions."""
    sp, rx, summary = load_seed_mechanism("GRI-Mech3.0-N", _db())
    assert len(sp) > 0, "seed species empty"
    assert len(rx) > 0, "seed reactions empty (the B1 stub)"
    assert summary["n_species"] > 0 and summary["n_reactions"] > 0
    assert summary["resolution"].startswith("alias")
    assert summary["n_reactions_dropped_missing_species"] == 0


# ---------------------------------------------------------------------------
# Fixture builder (real adjlists) for the loud-behaviour tests
# ---------------------------------------------------------------------------
def _build_kinetics_db(kpath, species, reactions):
    """Build a minimal kinetics.db. species: list of (label, adjlist).
    reactions: list of dicts {label, degeneracy, reversible,
    reactants:[label], products:[label], arrhenius:[(A_val,A_unit,Ea_val,Ea_unit,n,T0)]}."""
    conn = sqlite3.connect(str(kpath))
    cur = conn.cursor()
    cur.execute("CREATE TABLE kinetics_libraries_table (id INTEGER PRIMARY KEY, name TEXT)")
    cur.execute("INSERT INTO kinetics_libraries_table VALUES (1, 'FixLib')")
    cur.execute("""CREATE TABLE kinetics_library_dictionary_table
                   (id INTEGER PRIMARY KEY, library_id INTEGER, label TEXT, adjacency_list TEXT)""")
    for i, (label, adj) in enumerate(species, start=1):
        cur.execute("INSERT INTO kinetics_library_dictionary_table VALUES (?, 1, ?, ?)",
                    (i, label, adj))
    cur.execute("""CREATE TABLE kinetics_library_reactions_table
                   (id INTEGER PRIMARY KEY, library_id INTEGER, label TEXT,
                    degeneracy FLOAT, short_description TEXT, long_description TEXT,
                    rank INTEGER, allow_max_rate_violation BOOLEAN, reversible BOOLEAN,
                    elementary_high_p BOOLEAN, duplicate BOOLEAN)""")
    cur.execute("""CREATE TABLE kinetics_library_reaction_species_table
                   (id INTEGER PRIMARY KEY, library_reaction_id INTEGER,
                    species_label TEXT, role TEXT)""")
    cur.execute("""CREATE TABLE kinetics_arrhenius_table
                   (id INTEGER PRIMARY KEY, library_reaction_id INTEGER,
                    family_rule_id INTEGER, family_training_reaction_id INTEGER,
                    kinetics_type TEXT, A_val FLOAT, A_unit TEXT, n FLOAT,
                    Ea_val FLOAT, Ea_unit TEXT, T0_val FLOAT, T0_unit TEXT,
                    Tmin_val FLOAT, Tmin_unit TEXT, Tmax_val FLOAT, Tmax_unit TEXT,
                    comment TEXT)""")
    rid = 10
    sid = 1
    for r in reactions:
        cur.execute("""INSERT INTO kinetics_library_reactions_table
                       (id, library_id, label, degeneracy, reversible)
                       VALUES (?, 1, ?, ?, ?)""",
                    (rid, r["label"], r.get("degeneracy", 1.0), r.get("reversible", 0)))
        for s in r["reactants"] + r["products"]:
            cur.execute("INSERT INTO kinetics_library_reaction_species_table "
                        "VALUES (?, ?, ?, ?)", (sid, rid, s,
                        "reactant" if s in r["reactants"] else "product"))
            sid += 1
        for row in r["arrhenius"]:
            A_val, A_unit, Ea_val, Ea_unit, n, T0 = row
            cur.execute("""INSERT INTO kinetics_arrhenius_table
                           (library_reaction_id, A_val, A_unit, n, Ea_val, Ea_unit, T0_val)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (rid, A_val, A_unit, n, Ea_val, Ea_unit, T0))
        rid += 1
    conn.commit()
    conn.close()


def _empty_thermo_db(tpath):
    tconn = sqlite3.connect(str(tpath))
    tconn.execute("CREATE TABLE thermo_libraries_view (id INTEGER PRIMARY KEY, label TEXT, name TEXT)")
    tconn.commit()
    tconn.close()


def _fixture_dbs(tmp_path, species, reactions):
    kpath = str(tmp_path / "kinetics.db")
    tpath = str(tmp_path / "thermo.db")
    _build_kinetics_db(kpath, species, reactions)
    _empty_thermo_db(tpath)
    return Databases.from_config({"paths": {"thermo": tpath, "kinetics": kpath}})


# ---------------------------------------------------------------------------
# Fix 2: no placeholder methane; missing species are skipped + counted
# ---------------------------------------------------------------------------
def test_missing_species_skipped_not_methane(tmp_path):
    species = [("H2", ADJ_H2), ("H", ADJ_H)]
    reactions = [
        {"label": "H2 -> H + MISSING", "reactants": ["H2"],
         "products": ["H", "MISSING"],
         "arrhenius": [(2.5e6, "cm^3/(mol*s)", 100.0, "cal/mol", 1.0, 1.0)]},
    ]
    db = _fixture_dbs(tmp_path, species, reactions)
    sp, rx, summary = load_seed_mechanism("FixLib", db)
    # 'MISSING' is not in the dictionary -> the reaction is SKIPPED + counted,
    # and NO placeholder methane species is created.
    assert summary["n_reactions_dropped_missing_species"] == 1
    assert len(rx) == 0
    labels = {s.label for s in sp}
    assert "MISSING" not in labels, "a placeholder species was created"
    for s in sp:
        f = s.molecule.get_formula()
        assert f != "CH4", f"a methane placeholder was injected ({f})"


# ---------------------------------------------------------------------------
# Fix 4: multi-band dropped + counted (no silent LIMIT 1)
# ---------------------------------------------------------------------------
def test_multiband_not_silently_limit1(tmp_path):
    species = [("H2", ADJ_H2), ("H", ADJ_H)]
    reactions = [
        {"label": "H2 -> H + H", "reactants": ["H2"], "products": ["H", "H"],
         "arrhenius": [
             (2.5e6, "cm^3/(mol*s)", 100.0, "cal/mol", 1.0, 1.0),
             (9.9e6, "cm^3/(mol*s)", 200.0, "cal/mol", 1.0, 1.0),
         ]},
    ]
    db = _fixture_dbs(tmp_path, species, reactions)
    sp, rx, summary = load_seed_mechanism("FixLib", db)
    assert summary["n_species"] == 2, f"species not parsed: {summary}"
    assert summary["n_reactions_dropped_multiband"] == 1
    assert summary["n_reactions"] == 0, "multi-band reaction was silently kept"
    assert len(rx) == 0


def test_single_band_reaction_loads(tmp_path):
    species = [("H2", ADJ_H2), ("H", ADJ_H)]
    reactions = [
        {"label": "H2 -> H + H", "reactants": ["H2"], "products": ["H", "H"],
         "arrhenius": [(2.5e6, "cm^3/(mol*s)", 100.0, "cal/mol", 1.0, 1.0)]},
    ]
    db = _fixture_dbs(tmp_path, species, reactions)
    sp, rx, summary = load_seed_mechanism("FixLib", db)
    assert summary["n_reactions"] == 1 and len(rx) == 1
    # A converted by 1e-6 (cm^3) and Ea converted from cal/mol
    rp = rx[0].rate_model
    assert rp.A == pytest.approx(2.5e6 * 1e-6)
    assert rp.Ea == pytest.approx(100.0 * 4.184)


# ---------------------------------------------------------------------------
# Fix 5: termolecular A by the ACTUAL unit (1e-12), not 1e-6
# ---------------------------------------------------------------------------
def test_termolecular_A_converted_by_1e_minus_12():
    row = {"A_val": 1.0e10, "A_unit": "cm^6/(mol^2*s)", "n": 2.0,
           "Ea_val": 100.0, "Ea_unit": "cal/mol", "T0_val": 1.0}
    rp = _flat_rate_param([row], True, 1.0)
    assert rp.A == pytest.approx(1.0e10 * 1e-12), (
        f"cm^6/(mol^2*s) must get 1e-12, got A={rp.A}")
    # the old substring heuristic gave 1e-6: prove that is NOT what happened
    assert rp.A != pytest.approx(1.0e10 * 1e-6)
    # cal/mol Ea converted to J/mol
    assert rp.Ea == pytest.approx(100.0 * 4.184)


def test_bimolecular_A_still_1e_minus_6():
    row = {"A_val": 2.5e6, "A_unit": "cm^3/(mol*s)", "n": 1.0,
           "Ea_val": 50.0, "Ea_unit": "cal/mol", "T0_val": 1.0}
    rp = _flat_rate_param([row], True, 1.0)
    assert rp.A == pytest.approx(2.5e6 * 1e-6)


# ---------------------------------------------------------------------------
# Gate physical-validity check (Fix 5 in gate_06.py)
# ---------------------------------------------------------------------------
def _write_profile(tmp_path, rows, header_cols):
    root = tmp_path / "run"
    prof = root / "profiles" / "reactor"
    prof.mkdir(parents=True)
    with open(prof / "time_series.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time[s]"] + [f"{c}_molefrac" for c in header_cols])
        for r in rows:
            w.writerow(r)
    return str(root)


def test_physical_validity_passes_physical_profile(tmp_path):
    root = _write_profile(
        tmp_path,
        [[0.0, 0.6, 0.4, 0.0, 0.0, 0.0], [1.0, 0.3, 0.7, 0.0, 0.0, 0.0]],
        ["H2", "O2", "H2O", "H", "OH"])
    out = G.check_physical_validity(root)
    assert out["status"] == "PASS"
    assert out["bad_values"] == []
    assert out["row_sum_errors"] == []


def test_physical_validity_fails_out_of_range(tmp_path):
    # a non-physical blow-up: a mole fraction of 250 (the recorded defect)
    root = _write_profile(
        tmp_path,
        [[0.0, 0.6, 0.4, 0.0, 0.0, 0.0], [1.0, 250.0, 0.2, 0.0, 0.0, 0.0]],
        ["H2", "O2", "H2O", "H", "OH"])
    out = G.check_physical_validity(root)
    assert out["status"] == "FAIL"
    assert out["bad_values"], "out-of-range values must be recorded"
    bad = {b["species"]: b["value"] for b in out["bad_values"]}
    assert bad.get("H2") == pytest.approx(250.0)


def test_physical_validity_fails_bad_row_sum(tmp_path):
    # all values in [0,1] but the row does not sum to 1 (open/broken state)
    root = _write_profile(
        tmp_path,
        [[0.0, 0.6, 0.4, 0.0, 0.0, 0.0], [1.0, 0.2, 0.2, 0.1, 0.0, 0.0]],
        ["H2", "O2", "H2O", "H", "OH"])
    out = G.check_physical_validity(root)
    assert out["status"] == "FAIL"
    assert out["row_sum_errors"], "row-sum errors must be recorded"
    assert out["bad_values"] == []  # values in range; only the sum is off


def test_physical_validity_fails_non_finite(tmp_path):
    # a nan (blow-up / overflowed rate) must FAIL even though it is not
    # caught by the [-eps, 1+eps] or row-sum comparisons alone.
    root = tmp_path / "run"
    prof = root / "profiles" / "reactor"
    prof.mkdir(parents=True)
    with open(prof / "time_series.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time[s]", "H2_molefrac", "O2_molefrac"])
        w.writerow([0.0, 0.6, 0.4])
        w.writerow([1.0, "nan", 0.4])
    out = G.check_physical_validity(str(root))
    assert out["status"] == "FAIL"
    assert any(b.get("reason", "").startswith("non-finite")
               for b in out["bad_values"]), f"nan not flagged: {out['bad_values']}"


def test_physical_validity_missing_file_fails(tmp_path):
    out = G.check_physical_validity(str(tmp_path / "nope"))
    assert out["status"] == "FAIL"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
