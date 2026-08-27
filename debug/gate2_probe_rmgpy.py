#!/usr/bin/env python
"""Probe RMG-Py thermo/kinetics entry shapes (run in rmg_env). One-off, not committed as deliverable."""
import sys, json
import rmgpy
print("rmgpy file:", rmgpy.__file__)
from rmgpy.data.thermo import ThermoDatabase
from rmgpy.data.kinetics import KineticsDatabase

THERMO_DIR = "/home/jackson/rmgpu/RMG-database/input/thermo"
KIN_DIR = "/home/jackson/rmgpu/RMG-database/input/kinetics"

tdb = ThermoDatabase()
tdb.load(THERMO_DIR)
print("thermo libraries loaded:", len(tdb.libraries))

lib = tdb.libraries["primaryThermoLibrary"]
print("primaryThermoLibrary entries:", len(lib.entries), "type:", type(lib.entries))
e = lib.entries["H2"]
print("entry type:", type(e))
print("entry attrs:", [a for a in dir(e) if not a.startswith("__")])
print("label:", e.label)
print("molecule type:", type(e.molecule))
# molecule to adjacency list
try:
    al = e.molecule.to_adjacency_list()
    print("to_adjacency_list repr:")
    print(repr(al))
except Exception as ex:
    print("to_adjacency_list err:", ex)
print("thermo type:", type(e.thermo))
print("thermo attrs:", [a for a in dir(e.thermo) if not a.startswith("__")])
# show a ThermoData
td = e.thermo
for attr in ["Tdata","Cpdata","H298","S298","E0","Cp0","CpInf"]:
    try:
        v = getattr(td, attr)
        print(f"  thermo.{attr} = {v!r} (type {type(v).__name__})")
    except Exception as ex:
        print(f"  thermo.{attr} err {ex}")

# Find a NASA entry in BurcatNS
b = tdb.libraries["BurcatNS"]
for e2 in list(b.entries.values())[:300]:
    if type(e2.thermo).__name__ in ("NASA", "NasaPolynomial"):
        print("\nNASA example:", e2.label, type(e2.thermo).__name__)
        for attr in ["Tmin","Tmax","H298","S298","E0","Cp0","CpInf","high","low"]:
            try:
                print(f"   .{attr} = {getattr(e2.thermo, attr)!r}")
            except Exception as ex:
                print(f"   .{attr} err {ex}")
        break

print("\n=== KINETICS ===")
kdb = KineticsDatabase()
kdb.load(KIN_DIR, families=[], depositories=[])
print("kinetics libraries:", len(kdb.libraries))
kl = kdb.libraries["primaryH2O2"]
print("primaryH2O2 entries:", len(kl.entries), "type:", type(kl.entries))
k = list(kl.entries.values())[0]
print("kinetics entry type:", type(k))
print("kinetics entry attrs:", [a for a in dir(k) if not a.startswith("__")])
print("label:", k.label)
print("reaction type:", type(k.reaction))
rate = k.reaction.rate
print("rate type:", type(rate).__name__)
print("rate attrs:", [a for a in dir(rate) if not a.startswith("__")])
for attr in ["A","n","Ea","T0","Tmin","Tmax"]:
    try:
        v = getattr(rate, attr)
        print(f"  rate.{attr} = {v!r} (type {type(v).__name__})")
    except Exception as ex:
        print(f"  rate.{attr} err {ex}")
# eval k at 300,1bar
R = 8.31446261815324
P = 1e5
try:
    k300 = rate.get_rate_coefficient(300.0, P)
    k1000 = rate.get_rate_coefficient(1000.0, P)
    print("k(300,1bar) =", k300, " k(1000,1bar) =", k1000)
except Exception as ex:
    print("eval err:", ex)
