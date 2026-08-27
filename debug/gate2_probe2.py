#!/usr/bin/env python
"""Probe 2: Entry.molecule / data field types. Run in rmg_env."""
import logging
logging.disable(logging.CRITICAL)
import rmgpy
from rmgpy.data.thermo import ThermoDatabase
from rmgpy.data.kinetics import KineticsDatabase
from rmgpy.data.base import Entry
import inspect

# Where does the molecule come from?
print("Entry has molecule attr?", hasattr(Entry, "molecule"))
print("Entry fields (from __init__/slots):")
try:
    src = inspect.getsource(Entry.__init__)
    print(src[:2000])
except Exception as ex:
    print("no src", ex)

THERMO_DIR = "/home/jackson/rmgpu/RMG-database/input/thermo"
KIN_DIR = "/home/jackson/rmgpu/RMG-database/input/kinetics"
tdb = ThermoDatabase()
tdb.load(THERMO_DIR)
lib = tdb.libraries["primaryThermoLibrary"]
e = lib.entries["H2"]
print("\n=== thermo Entry H2 ===")
print("data type:", type(e.data).__name__)
print("data_count:", e.data_count)
td = e.data
for attr in ["Tdata","Cpdata","H298","S298","E0","Cp0","CpInf","Tmin","Tmax"]:
    try:
        print(f"  .{attr} = {getattr(td,attr)!r}")
    except Exception as ex:
        print(f"  .{attr} err {ex}")
# does the entry carry molecule anywhere?
print("entry keys:", [a for a in vars(e).keys()])
# try to get molecule via the species dict of the database
# ThermoDatabase may have .species or entries store molecule in e.data?
print("does tdb have species?", hasattr(tdb, "species"))

# NASA example from BurcatNS with segment structure
b = tdb.libraries["BurcatNS"]
for e2 in list(b.entries.values())[:400]:
    if type(e2.data).__name__ in ("NASA","NasaPolynomial"):
        print("\n=== NASA example:", e2.label, type(e2.data).__name__, "===")
        for attr in ["Tmin","Tmax","H298","S298","E0","Cp0","CpInf","high","low"]:
            try:
                v = getattr(e2.data, attr)
                print(f"  .{attr} = {v!r}")
            except Exception as ex:
                print(f"  .{attr} err {ex}")
        break

print("\n=== KINETICS ===")
kdb = KineticsDatabase()
kdb.load(KIN_DIR, families=[], depositories=[])
kl = kdb.libraries["primaryH2O2"]
print("primaryH2O2 entries:", len(kl.entries), type(kl.entries))
k = list(kl.entries.values())[0]
print("entry type:", type(k).__name__)
print("label:", k.label)
print("entry vars:", [a for a in vars(k).keys()])
rxn = k.data  # in kinetics, data is the Reaction
print("data type:", type(rxn).__name__)
rate = rxn.rate
print("rate type:", type(rate).__name__)
print("rate vars:", [a for a in vars(rate).keys()])
for attr in ["A","n","Ea","T0","Tmin","Tmax","high","low","alpha"]:
    try:
        print(f"  rate.{attr} = {getattr(rate,attr)!r}")
    except Exception as ex:
        print(f"  rate.{attr} err {ex}")
