"""Settle the unit of RMG-Py get_rate_coefficient. Run in rmg_env.

A reaction with A=(1e13,'cm^3/(mol*s)'), n=0, Ea=0, T0=1 has
get_rate_coefficient(T) == A in whatever unit the kinetics object stores.
  -> ~1e13      => CGS (cm^3/mol*s)
  -> ~1e7       => SI (m^3/mol*s)
Also dump O(T)+H2 (A=38700, n=2.7, Ea=6260 cal/mol) k(300) in both units.
"""
from __future__ import annotations
import sys

DB = "/home/jackson/rmgpu/RMG-database/input"

from rmgpy.data.rmg import RMGDatabase
_ = RMGDatabase()
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.kinetics import Arrhenius

kdb = KineticsDatabase()
kdb.load_recommended_families(DB + "/kinetics/families/recommended.py")
kdb.load_families(DB + "/kinetics/families", families="none", depositories=None)
kdb.load_libraries(DB + "/kinetics/libraries", libraries=["GRI-Mech3.0-N"])

def val(x):
    if x is None:
        return 0.0
    if hasattr(x, "value"):
        try:
            return float(x.value)
        except Exception:
            pass
    return float(x)

from rmgpy.quantity import ScalarQuantity

def q(x):
    """Return (value_in_stored_units, units-string, value_si)."""
    if x is None:
        return 0.0, "dimensionless", 0.0
    if hasattr(x, "units"):
        v = x.value
        v = float(v) if not hasattr(v, "value") else float(v.value)
        return v, str(x.units), float(x.value_si)
    return float(x), "dimensionless", float(x)

found_A1e13 = False
found_OH2 = False
for rxn in kdb.libraries["GRI-Mech3.0-N"].get_library_reactions():
    k = rxn.kinetics
    if not isinstance(k, Arrhenius):
        continue
    A, Aunits, A_si = q(k.A)
    n = val(k.n)
    Ea, Eaunits, Ea_si = q(k.Ea)
    if abs(A - 1e13) < 1e9 and abs(n) < 1e-9 and abs(Ea) < 1.0:
        k300 = k.get_rate_coefficient(300.0)
        print(f"A=1e13[{Aunits}] n=0 Ea=0 label={rxn.label!r}: k(300)={k300:.6e}  (1e13->CGS, 1e7->SI)")
        found_A1e13 = True
        if found_A1e13 and found_OH2:
            break
    if rxn.label and "O(T) + H2" in rxn.label and "<=>" in rxn.label:
        k300 = k.get_rate_coefficient(300.0)
        k600 = k.get_rate_coefficient(600.0)
        print(f"O(T)+H2 label={rxn.label!r}: A={A!r}[{Aunits}] n={n} Ea={Ea!r}[{Eaunits}]")
        print(f"   get_rate_coefficient(300)={k300:.6e}  (600)={k600:.6e}")
        found_OH2 = True
