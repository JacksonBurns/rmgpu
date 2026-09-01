"""Compare reactant-only subgraph match vs the full RMG verdict (which
includes recipe+product check). If they agree on all 322 (family, reaction)
pairs, a reactant-only matcher is faithful enough for the step-03 test set."""
import json, sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule import group as G

REF = '/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step03_templates_reference.json'
VERD = '/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step03_match_verdicts.json'
d = json.load(open(REF)); v = json.load(open(VERD))
G.load_atomtype_tree(d['atomtype_tree'])

fam_tree = d['families']
verdicts = v['verdicts']
rxns = v['reactions']

def slot_groups(adj):
    # each slot is a dict with 'components': list of adjlist strings
    return [G.parse_group_adjlist_full(a) for a in adj['components']]

def reactant_only_matches(fam_name, rxn):
    """Does fam's forward template reactant-subgraph-match the reaction's
    reactants? Returns True if EVERY forward-template slot can be matched to
    some reactant (greedy, ignoring which reactant) OR using RMG's per-slot
    reactant assignment. We emulate RMG: template has len == len(reactants)?
    For 2-reactant families each slot maps to one reactant."""
    fwd = fam_tree[fam_name]['forward_template']
    # fwd is a list of slot dicts; each slot has 'components'
    react_smiles = rxn['reactants']
    mols = [Molecule(smiles=s) for s in react_smiles]
    # The template slots: number of slots should equal number of reactants
    # for bimolecular; for unimolecular intra families it's 1 reactant.
    slots = fwd
    if len(slots) != len(mols):
        # cannot 1:1 assign -> treat as no (RMG would fail reactant match)
        return False
    for slot, mol in zip(slots, mols):
        gs = slot_groups(slot)
        # try each component (LogicOr)
        ok = False
        for g in gs:
            if G.match_group(mol, g):
                ok = True; break
        if not ok:
            return False
    return True

agree = 0
disagree = 0
disag_list = []
for rec in rxns:
    ci, ri, own = rec['ci'], rec['ri'], rec['family']
    for fl in v['families']:
        key = '%s|%d|%d' % (fl, ci, ri)
        ref_full = bool(verdicts[key]['matched'])
        ro = reactant_only_matches(fl, rec)
        if ref_full == ro:
            agree += 1
        else:
            disagree += 1
            if len(disag_list) < 15:
                disag_list.append((key, 'full=%s reactonly=%s' % (ref_full, ro)))

print("total pairs:", agree + disagree)
print("agree:", agree, " disagree:", disagree)
for x in disag_list:
    print("  DISAGREE", x[0], x[1])
