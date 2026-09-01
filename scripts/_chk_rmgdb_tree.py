import sqlite3, json, os

DB = '/home/jackson/rmgpu/rmgdb/db/kinetics.db'
con = sqlite3.connect(DB)
cur = con.cursor()

# table names
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print('tables:', tables)

cur.execute("SELECT count(*) FROM kinetics_families_table")
print('families in rmgdb:', cur.fetchone()[0])
cur.execute("SELECT count(*) FROM kinetics_family_groups_table")
print('groups in rmgdb:', cur.fetchone()[0])
if 'kinetics_family_groups_tree_table' in tables:
    cur.execute("SELECT count(*) FROM kinetics_family_groups_tree_table")
    print('tree edges in rmgdb:', cur.fetchone()[0])
    cur.execute("PRAGMA table_info(kinetics_family_groups_tree_table)")
    print('tree cols:', [r[1] for r in cur.fetchall()])
else:
    print('NO kinetics_family_groups_tree_table')
