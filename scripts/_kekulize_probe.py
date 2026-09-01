#!/usr/bin/env python3
"""Reproduce the EXACT rmgpu benzene+H merged state, print aromatic flags +
bond orders at each stage, and test which kekulize variant resolves the 1.5
bonds. Decides the fix for Bug C."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.core.recipe import (ReactionRecipe, _merge_molecules, _re_aromatize,
                               _kekulize_piece, _is_benzene)
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

print('reactant aromatic flags:')
for s in structures:
    print('  ', s.get_formula(), [a.GetIsAromatic() for a in s._rdkit.GetAtoms()])

merged = _merge_molecules(structures)
print('merged flags:', [a.GetIsAromatic() for a in merged.GetAtoms()])
print('merged orders:', [b.GetBondTypeAsDouble() for b in merged.GetBonds()])
_re_aromatize(merged)
print('re-aro flags:', [a.GetIsAromatic() for a in merged.GetAtoms()])
print('re-aro orders:', [b.GetBondTypeAsDouble() for b in merged.GetBonds()])

recipe._apply(merged, True, True)
print('post-recipe flags:', [a.GetIsAromatic() for a in merged.GetAtoms()])
print('post-recipe orders:', [b.GetBondTypeAsDouble() for b in merged.GetBonds()])
print('post-recipe benzene(1.5) bonds:',
      sum(1 for b in merged.GetBonds() if _is_benzene(b.GetBondTypeAsDouble())))

m = Chem.Mol(merged)
try:
    Chem.Kekulize(m, clearAromaticFlags=True)
    print('Kekulize(merged) OK orders:', [b.GetBondTypeAsDouble() for b in m.GetBonds()])
except Exception as e:
    print('Kekulize(merged) FAIL:', type(e).__name__, e)
    print('   1.5 left:', sum(1 for b in m.GetBonds() if _is_benzene(b.GetBondTypeAsDouble())))
    # try kekulize on a copy where ring atoms are forced aromatic
    m2 = Chem.Mol(merged)
    ring_atoms = [a.GetIdx() for a in m2.GetAtoms() if a.GetIsAromatic()]
    print('   aromatic-flagged atoms:', ring_atoms)
    for i in range(len(m2.GetAtoms())):
        pass
    # force all ring carbons aromatic
    for a in m2.GetAtoms():
        if a.GetSymbol() == 'C':
            a.SetIsAromatic(True)
    try:
        Chem.Kekulize(m2, clearAromaticFlags=True)
        print('   Kekulize(force-aromatic) OK orders:',
              [b.GetBondTypeAsDouble() for b in m2.GetBonds()])
    except Exception as e2:
        print('   Kekulize(force-aromatic) FAIL:', type(e2).__name__, e2)
