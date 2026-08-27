#!/usr/bin/env python
"""Probe 3: raw (value, unit) on both sides for primaryThermoLibrary + primaryH2O2. Run in rmg_env."""
import logging
logging.disable(logging.CRITICAL)
import sqlite3, json
from rmgpy.data.thermo import ThermoDatabase
from rmgpy.data.kinetics import KineticsDatabase

THERMO_DIR = "/home/jackson/rmgpu/RMG-database/input/thermo"
KIN_DIR = "/home/jackson/rmgpu/RMG-database/input/kinetics"
TDB = "/home/jackson/rmgpu/rmgdb/db/thermo.db"
KDB = "/home/jackson/rmgpu/rmgdb/db/kinetics.db"

def qty(x):
    if x is None:
        return None
    if hasattr(x, "value") and hasattr(x, "units"):
        v = x.value
        try:
            v = [float(t) for t in v]
        except TypeError:
            try:
                v = float(v)
            except Exception:
                v = repr(v)
        return {"value": v, "unit": str(x.units)}
    return {"raw": repr(x), "type": type(x).__name__}

tdb = ThermoDatabase(); tdb.load(THERMO_DIR)
lib = tdb.libraries["primaryThermoLibrary"]

print("### THERMO primaryThermoLibrary : RMG-Py side ###")
for lbl in ["H2", "H", "CO3s1"]:
    e = lib.entries[lbl]; d = e.data
    out = {"type": type(d).__name__}
    if type(d).__name__ == "ThermoData":
        out["Tdata"] = qty(d.Tdata); out["Cpdata"] = qty(d.Cpdata)
        out["H298"] = qty(d.H298); out["S298"] = qty(d.S298)
        out["E0"] = qty(d.E0); out["Cp0"] = qty(d.Cp0); out["CpInf"] = qty(d.CpInf)
    print(lbl, json.dumps(out))
# to_si check
e = lib.entries["H"]; d = e.data
print("H298 raw:", qty(d.H298), " to_si:", d.H298.to_si() if hasattr(d.H298,'to_si') else 'no tosi', " val*4184:", float(d.H298.value)*4184.0)

print("\n### THERMO primaryThermoLibrary : rmgdb side (view raw) ###")
conn = sqlite3.connect(TDB); cur = conn.cursor()
for lbl in ["H2", "H", "CO3s1"]:
    row = cur.execute("SELECT H298,H298_unit,S298,S298_unit,Tdata_1,Tdata_7,Cpdata_1,Cpdata_unit,E0,E0_unit FROM thermo_libraries_view WHERE name='primaryThermoLibrary' AND label=? LIMIT 1", (lbl,)).fetchone()
    print(lbl, row)
conn.close()

print("\n### KINETICS primaryH2O2 : RMG-Py side ###")
kdb = KineticsDatabase(); kdb.load(KIN_DIR, families=[], depositories=[])
kl = kdb.libraries["primaryH2O2"]
def rate_fields(r):
    out = {"type": type(r).__name__}
    for a in ["A","n","Ea","T0","Tmin","Tmax","arrheniusHigh","arrheniusLow","alpha","T1","T2","T3"]:
        if hasattr(r, a):
            v = getattr(r, a)
            out[a] = qty(v)
    if type(r).__name__ in ("ThirdBody",):
        out["arrheniusLow"] = rate_fields(r.arrheniusLow)
    if type(r).__name__ == "Lindemann":
        out["arrheniusHigh"] = rate_fields(r.arrheniusHigh); out["arrheniusLow"] = rate_fields(r.arrheniusLow)
    if type(r).__name__ == "Troe":
        out["arrheniusHigh"] = rate_fields(r.arrheniusHigh); out["arrheniusLow"] = rate_fields(r.arrheniusLow)
    if type(r).__name__ == "PDepArrhenius":
        out["pressures"] = [qty(p) for p in r.pressures] if hasattr(r,'pressures') else 'no'
    return out
for k in list(kl.entries.values())[:4]:
    print(k.label, "->", json.dumps(rate_fields(k.data))[:400])
print("k(300,1e5) first entry:", kl.entries[list(kl.entries)[0]].data.get_rate_coefficient(300.0, 1e5))

print("\n### KINETICS primaryH2O2 : rmgdb side (raw) ###")
conn = sqlite3.connect(KDB); cur = conn.cursor()
rows = cur.execute("""
SELECT r.id, r.label FROM kinetics_library_reactions_table r
JOIN kinetics_libraries_table k ON r.library_id=k.id
WHERE k.name='primaryH2O2' ORDER BY r.id LIMIT 4
""").fetchall()
for rid, label in rows:
    print("reaction", rid, "label:", repr(label))
    print("  arr:", cur.execute("SELECT A_val,A_unit,n,Ea_val,Ea_unit,T0_val,Tmin_val,Tmax_val FROM kinetics_arrhenius_table WHERE library_reaction_id=?",(rid,)).fetchall())
    print("  thirdbody:", cur.execute("SELECT low_A_val,low_A_unit,low_n,low_Ea_val,low_Ea_unit FROM kinetics_third_body_table WHERE library_reaction_id=?",(rid,)).fetchall())
    print("  lind:", cur.execute("SELECT high_A_val,high_A_unit,high_n,high_Ea_val,high_Ea_unit,low_A_val,low_A_unit,low_n,low_Ea_val,low_Ea_unit FROM kinetics_lindemann_table WHERE library_reaction_id=?",(rid,)).fetchall())
    print("  pdep:", cur.execute("SELECT id FROM kinetics_pdep_arrhenius_table WHERE library_reaction_id=?",(rid,)).fetchall())
conn.close()
