import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdkit import Chem
from rmgpu.core.recipe import (ReactionRecipe, _merge_molecules, _re_aromatize,
                               _is_benzene, VALENCES, _lone_pairs)

def six_rings(mol):
    n = mol.GetNumAtoms()
    adj = {i: [] for i in range(n)}
    for b in mol.GetBonds():
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[a].append(c); adj[c].append(a)
    rings = set()
    for start in range(n):
        for n1 in adj[start]:
            if n1 < start: continue
            for n2 in adj[n1]:
                if n2 == start or n2 <= start: continue
                for n3 in adj[n2]:
                    if n3 in (start, n1) or n3 <= start: continue
                    for n4 in adj[n3]:
                        if n4 in (start, n1, n2) or n4 <= start: continue
                        for n5 in adj[n4]:
                            if n5 in (start, n1, n2, n3) or n5 <= start: continue
                            if start in adj[n5]:
                                rings.add(frozenset((start, n1, n2, n3, n4, n5)))
    return [sorted(r) for r in rings]

def bond_info(mol, a1, a2):
    dp = True; dr = False
    for a in (a1, a2):
        atom = mol.GetAtomWithIdx(a)
        occupied = 0; uncertain = 0
        for bond in atom.GetBonds():
            order = bond.GetBondTypeAsDouble()
            if abs(round(order) - order) < 1e-9:
                occupied += int(round(order))
            elif _is_benzene(order):
                occupied += 1; uncertain += 1
        occupied += atom.GetNumRadicalElectrons()
        occupied += 2 * _lone_pairs(atom)
        available = VALENCES.get(atom.GetSymbol(), 4) - occupied
        if available < 0:
            return None
        if available == 0:
            dp = False
        elif available == 1 and uncertain == 1:
            dr = True
    return dp, dr

def set_bond_order(mol, a1, a2, order):
    b = mol.GetBondBetweenAtoms(a1, a2)
    b.SetBondType(Chem.BondType.SINGLE if order == 1.0 else
                  Chem.BondType.DOUBLE if order == 2.0 else Chem.BondType.TRIPLE)

def dof_kekulize(mol):
    for ring in six_rings(mol):
        endo = {}
        for i in range(6):
            a, b = ring[i], ring[(i+1) % 6]
            endo[frozenset((a, b))] = mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble()
        unresolved = {b for b, o in endo.items() if _is_benzene(o)}
        if not unresolved:
            continue
        it = 0; maxiter = 2 * len(unresolved) + 6
        while unresolved and it < maxiter:
            it += 1
            assigned = None
            for b in list(unresolved):
                a1, a2 = tuple(b)
                info = bond_info(mol, a1, a2)
                if info is None:
                    return False
                dp, dr = info
                if dp and dr:
                    assigned = (b, 2.0); break
                elif not dp:
                    assigned = (b, 1.0); break
            if assigned is None:
                b = next(iter(unresolved))
                a1, a2 = tuple(b)
                info = bond_info(mol, a1, a2)
                if info is None:
                    return False
                assigned = (b, 2.0 if info[0] else 1.0)
            b, order = assigned
            a1, a2 = tuple(b)
            set_bond_order(mol, a1, a2, order)
            endo[b] = order
            unresolved.discard(b)
        if unresolved:
            return False
        for a in ring:
            mol.GetAtomWithIdx(a).SetIsAromatic(False)
    return True

# Build the post-recipe benzene+H structure
from rmgpu.core import enumeration as enum
import json
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'gates', 'baselines', 'job05', 'step02_products_reference.json')
data = json.load(open(BASE))
case = [c for c in data['cases'] if c['family']=='R_Addition_MultipleBond' and c['reactant_smiles']==['C1=CC=CC=C1','[H]']][0]
match = enum.RecordedMatcher(case)
structures = list(match.yield_applications())[0]
merged = _merge_molecules(structures)
_re_aromatize(merged)
recipe = ReactionRecipe.from_data(case['family_meta']['recipe'])
recipe._apply(merged, True, True)
print("before kekulize orders:", sorted(b.GetBondTypeAsDouble() for b in merged.GetBonds()))
ok = dof_kekulize(merged)
print("dof_kekulize ok:", ok)
print("after kekulize orders:", sorted(b.GetBondTypeAsDouble() for b in merged.GetBonds()))
smi = Chem.MolToSmiles(merged, canonical=True)
print("rmgpu kekulized:", smi)
# RMG recorded
for p in case['reactions'][0]['products']:
    from rmgpu.molecule.molecule import Molecule
    rm = Molecule.from_adjacency_list(p)
    print("RMG recorded    :", Chem.MolToSmiles(rm._rdkit, canonical=True))
