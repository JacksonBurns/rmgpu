"""Full parity check: my matcher's per-slot subgraph mapping counts (summed
across OR components, exactly as RMG _match_reactant_to_template does) vs the
recorded reference n_mappings."""
import json, sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule import group as G

p3 = '/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step03_templates_reference.json'
d3 = json.load(open(p3))

ok = 0
bad = []
for c in d3['match_cases']:
    fam = c['family']
    smiles = c['reactant_smiles']
    n_tr = c['n_template_slots']
    # build molecules
    mols = [Molecule(smiles=s) for s in smiles]
    # for each (reactant, slot) pair, the reference has a per_slot entry.
    # My count = sum over slot components of match_group.
    # But careful: reference per_slot entries iterate ri over reactants, ti over slots.
    # For each such entry, ref n_mappings is for reactant ri vs slot ti's WHOLE item
    # (LogicNode union). My count = sum over slot_groups_adj[ti] components.
    for ps in c['per_slot']:
        ri = ps['reactant_index']
        ti = ps['template_slot']
        ref_n = ps['n_mappings']
        my_n = 0
        for comp_adj in c['slot_groups_adj'][ti]:
            g = G.parse_group_adjlist_full(comp_adj)
            my_n += len(G.match_group(mols[ri], g))
        status = 'OK' if my_n == ref_n else f'MISMATCH mine={my_n} ref={ref_n}'
        if my_n == ref_n:
            ok += 1
        else:
            bad.append((fam, smiles[ri], ti, my_n, ref_n))
        print(f"  {fam:28} {smiles[ri]:20} slot{ti}: {status}")
print(f"\nTotal pairs: {ok}/{ok+len(bad)}")
print("Mismatches:", len(bad))
for b in bad:
    print("  ", b)
