"""Probe the group matcher against the recorded RMG-Py ground truth."""
import json, sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule import group as G
from rmgpu.molecule.atomtype import assign_atom_types

p3 = '/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step03_templates_reference.json'
d3 = json.load(open(p3))
mc = d3['match_cases']

# Check atomtype_labels ordering consistency (AddHs determinism)
m = Molecule(smiles='CC')
e1 = m._with_explicit_h()
seq = [a.GetSymbol() for a in e1.GetAtoms()]
types = assign_atom_types(m)
print("CC explicit seq:", seq)
print("CC atomtypes   :", types, "  len match:", len(seq)==len(types))
e2 = m._with_explicit_h()
seq2 = [a.GetSymbol() for a in e2.GetAtoms()]
print("AddHs deterministic (same order twice):", seq == seq2)

print()
def run_case(c, verbose=False):
    slots = c['slot_groups_adj']
    smiles = c['reactant_smiles']
    tot_mine, tot_ref = 0, 0
    mismatch = 0
    for s in c['per_slot']:
        ri, ti, ref_n = s['reactant_index'], s['template_slot'], s['n_mappings']
        mol = Molecule(smiles=smiles[ri])
        mine = 0
        for adj in slots[ti]:
            grp = G.parse_group_adjlist_full(adj)
            mine += len(G.match_group(mol, grp))
        tot_mine += mine
        tot_ref += ref_n
        if mine != ref_n:
            mismatch += 1
            if verbose:
                print("  MISMATCH r%d slot%d: mine=%d ref=%d  (%s x %s)" % (
                    ri, ti, mine, ref_n, smiles[ri],
                    slots[ti][0][:40].replace('\n',' ')))
    return tot_mine, tot_ref, mismatch

allok = True
for c in mc:
    mine, ref, mm = run_case(c)
    status = "OK " if mine == ref else "FAIL"
    if mine != ref: allok = False
    print("%s %-26s %-22s mine=%-4d ref=%-4d mismatch_pairs=%d" % (
        status, c['family'], ','.join(c['reactant_smiles']), mine, ref, mm))
print()
print("ALL CASES MATCH:", allok)
