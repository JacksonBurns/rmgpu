import json, sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule import group as G
p3 = '/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step03_templates_reference.json'
d3 = json.load(open(p3))
mc = d3['match_cases']
targets = [('R_Addition_MultipleBond', ['C=C', '[CH3]']),
           ('intra_H_migration', ['C[CH]CCC']),
           ('Intra_ene_reaction', ['C[CH]C1=CC=CC=C1'])]
for fam, smi in targets:
    c = next(x for x in mc if x['family']==fam and x['reactant_smiles']==smi)
    slots = c['slot_groups_adj']
    print("\n\n############", fam, smi, "slots:", c['n_template_slots'])
    for s in c['per_slot']:
        ri, ti, ref_n = s['reactant_index'], s['template_slot'], s['n_mappings']
        if ref_n==0 and ti>=len(slots): continue
        mol = Molecule(smiles=smi[ri])
        for ci, adj in enumerate(slots[ti]):
            grp = G.parse_group_adjlist_full(adj)
            n = len(G.match_group(mol, grp))
            mark = "" 
            if ci==0:  # only total on first
                pass
            print("  r%d slot%d comp%d ref_slot_n=%d mine_comp=%d  adj=%s" % (
                ri, ti, ci, ref_n, n, repr(adj[:70])))
        # print full adj of each comp for this slot
        print("   --- slot%d full adjlists:" % ti)
        for ci, adj in enumerate(slots[ti]):
            print("   comp%d: %r" % (ci, adj))
