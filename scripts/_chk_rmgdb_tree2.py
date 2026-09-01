import sqlite3, sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.core.family import _parse_groups_file, _load_tree
import json

DB = '/home/jackson/rmgpu/rmgdb/db/kinetics.db'
FAMDIR = '/home/jackson/rmgpu/RMG-database/input/kinetics/families'
con = sqlite3.connect(DB)
cur = con.cursor()

ref = json.load(open('/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step04_families_reference.json'))
default = ref['default']


def rmgdb_edges(name):
    cur.execute(
        "SELECT count(*) FROM kinetics_family_groups_tree_table t "
        "WHERE t.parent_id IN (SELECT id FROM kinetics_family_groups_table "
        "  WHERE family_id=(SELECT id FROM kinetics_families_table WHERE name=?)) "
        "AND t.child_id IN (SELECT id FROM kinetics_family_groups_table "
        "  WHERE family_id=(SELECT id FROM kinetics_families_table WHERE name=?))",
        (name, name))
    return cur.fetchone()[0]


def file_edges(name):
    cap = _parse_groups_file('%s/%s/groups.py' % (FAMDIR, name))
    children, parent, top = _load_tree(cap['tree'])
    return sum(len(v) for v in children.values())


gap = []
for name in default:
    fe = file_edges(name)
    db = rmgdb_edges(name)
    if db != fe:
        gap.append((name, db, fe))
print('default families: %d; rmgdb-tree-gap: %d' % (len(default), len(gap)))
for g in gap:
    print('  %-45s rmgdb=%d file=%d' % g)
