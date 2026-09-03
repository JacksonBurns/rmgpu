#!/usr/bin/env python3
"""Probe rmgdb for the GRI-Mech3 library (seed-mechanism loading research)."""
import sqlite3

k = sqlite3.connect('/home/jackson/rmgpu/rmgdb/db/kinetics.db')
cur = k.cursor()

QID = "SELECT id FROM kinetics_library_reactions_table WHERE library_id=30"
Q = "SELECT id, label FROM kinetics_library_reactions_table WHERE library_id=30"
cur.execute(Q)
rxns = cur.fetchall()
print("GRI-Mech3 rxn rows:", len(rxns))

# rate types
cur.execute("""
SELECT a.kinetics_type, COUNT(*) FROM kinetics_arrhenius_table a
JOIN kinetics_library_reactions_table l ON a.library_reaction_id=l.id
WHERE l.library_id=30 GROUP BY a.kinetics_type""")
print("arrhenius-table rows by type:", cur.fetchall())
cur.execute("SELECT COUNT(*) FROM kinetics_pdep_arrhenius_table WHERE library_reaction_id IN (" + QID + ")")
print("pdep rows:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM kinetics_lindemann_table WHERE library_reaction_id IN (" + QID + ")")
print("lindemann rows:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM kinetics_troe_table WHERE library_reaction_id IN (" + QID + ")")
print("troe rows:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM kinetics_chebyshev_table WHERE library_reaction_id IN (" + QID + ")")
print("chebyshev rows:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM kinetics_third_body_table WHERE library_reaction_id IN (" + QID + ")")
print("third_body rows:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM kinetics_efficiencies_table WHERE library_reaction_id IN (" + QID + ")")
print("efficiency rows:", cur.fetchone()[0])
cur.execute("SELECT duplicate, reversible, COUNT(*) FROM kinetics_library_reactions_table WHERE library_id=30 GROUP BY 1,2")
print("duplicate/reversible:", cur.fetchall())
cur.execute("SELECT COUNT(*) FROM kinetics_library_reactions_table WHERE library_id=30 AND label IS NULL")
print("null labels:", cur.fetchone()[0])
# sample with a T range
cur.execute(Q + " LIMIT 5")
for rid, label in cur.fetchall():
    cur.execute("""SELECT kinetics_type, A_val, A_unit, n, Ea_val, Ea_unit, Tmin_val, Tmax_val
                   FROM kinetics_arrhenius_table WHERE library_reaction_id=?""", (rid,))
    print(label, "->", cur.fetchall())

# dictionary species
cur.execute("SELECT label, adjacency_list FROM kinetics_library_dictionary_table WHERE library_id=30 LIMIT 6")
for r in cur.fetchall():
    print("DICT:", r[0], "|", r[1][:60].replace("\n", " "))

# thermo: GRI-Mech3.0-N
t = sqlite3.connect('/home/jackson/rmgpu/rmgdb/db/thermo.db')
cur2 = t.cursor()
cur2.execute("SELECT name, COUNT(DISTINCT label) FROM thermo_libraries_table GROUP BY name")
for r in cur2.fetchall():
    print("THERMO LIB:", r)
cur2.execute("""SELECT name, COUNT(*) FROM thermo_data_table
                WHERE library_id IN (SELECT id FROM thermo_libraries_table WHERE name='GRI-Mech3.0-N')
                GROUP BY name""")
print("GRI thermo data rows:", cur2.fetchall()[:5], "total", cur2.execute(
    "SELECT COUNT(*) FROM thermo_data_table WHERE library_id IN (SELECT id FROM thermo_libraries_table WHERE name='GRI-Mech3.0-N')").fetchone()[0])
cur2.execute("""SELECT name, COUNT(*) FROM nasa_polynomial_table
                WHERE library_id IN (SELECT id FROM thermo_libraries_table WHERE name='GRI-Mech3.0-N')""")
print("GRI nasa rows:", cur2.fetchall())
