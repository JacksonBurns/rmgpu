#!/usr/bin/env python3
"""Decisive test for Bug C: does set-aromatic-flags + Chem.Kekulize resolve the
benzene+H cyclohexadienyl radical to the RMG-recorded Kekule form (doubles at
3=5, 4=6)? Compare canonical SMILES vs RMG's recorded product."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.core.recipe import (ReactionRecipe, _merge_molecules, _re_aromatize,
                               _kekulize_piece, _is_benzene, _connected_pieces)
from rmgpu.core import enumeration as enum
import json

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'gates', 'baselines', 'job05', 'step02_products_reference.json')
data = json.load(open(BASE))
case = [c for c in data['cases'] if c['family'] == 'R_Addition_MultipleBond'
        and c['reactant_smiles'] == ['C1=CC=CC=C1', '[H]']][0]
recipe = ReactionRecipe.from_data(case['family_meta']['recipe'])
match = enum.RecordedMatcher(case)
structures = list(match.yield_applications())[0]
merged = _merge_molecules(structures)
_re_aromatize(merged)
recipe._apply(merged, True, True)

# Build a Kekulized ring: force aromatic flags on atoms touching benzene bonds
# then Chem.Kekulize
m = Chem.Mol(merged)
for a in m.GetAtoms():
    for b in a.GetBonds():
        if _is_benzene(b.GetBondTypeAsDouble()):
            a.SetIsAromatic(True)
print('before kekulize: orders', sorted(b.GetBondTypeAsDouble() for b in m.GetBonds()))
try:
    Chem.Kekulize(m, clearAromaticFlags=True)
    ok = True
except Exception as e:
    ok = False
    print('kekulize FAIL:', type(e).__name__, e)
if ok:
    print('after kekulize: orders', sorted(b.GetBondTypeAsDouble() for b in m.GetBonds()))
    # find the double-bond pairs
    dbl = []
    for b in m.GetBonds():
        if b.GetBondTypeAsDouble() == 2.0:
            dbl.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
    print('double bonds:', dbl)
    smi = Chem.MolToSmiles(m, canonical=True)
    print('rmgpu kekulized SMILES:', smi)

# RMG recorded product
print('RMG recorded product adjlist:')
for p in case['reactions'][0]['products']:
    print(p)
    rmgmol = Molecule.from_adjacency_list(p)
    print('  RMG canon SMILES:', Chem.MolToSmiles(rmgmol._rdkit, canonical=True))
