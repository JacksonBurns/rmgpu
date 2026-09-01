"""Detailed per-pair mismatch probe."""
import json, sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule import group as G

p3 = '/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step03_templates_reference.json'
d3 = json.load(open(p3))
mc = d3['match_cases']

for c in mc:
    if not (c['family']=='H_Abstraction' and c['reactant_smiles']==['C','[H]']):
        continue
    slots = c['slot_groups_adj']
    print("case:", c['family'], c['reactant_smiles'], "  n_template_slots=", c['n_template_slots'])
    for s in c['per_slot']:
        ri, ti, ref_n = s['reactant_index'], s['template_slot'], s['n_mappings']
        mol = Molecule(smiles=c['reactant_smiles'][ri])
        print("\n  r%d(%s) slot%d label=%s  REF n=%d" % (ri, c['reactant_smiles'][ri], ti, s['template_label'], ref_n))
        for ci, adj in enumerate(slots[ti]):
            grp = G.parse_group_adjlist_full(adj)
            maps = G.match_group(mol, grp)
            first = adj.replace('\n',' ')[:60]
            print("     comp%d (%-45s) mine=%d  %s" % (ci, first, len(maps), ""))
