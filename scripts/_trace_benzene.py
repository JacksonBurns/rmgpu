#!/usr/bin/env python3
"""Trace the benzene+H recipe: where do 1.5 bonds come from, and why doesn't
kekulize resolve them?"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.core.recipe import ReactionRecipe, _merge_molecules, _re_aromatize, _kekulize_piece, _is_benzene
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
print('--- reactant pieces ---')
for s in structures:
    arom = [a.GetIdx() for a in s._rdkit.GetAtoms() if a.GetIsAromatic()]
    orders = sorted(b.GetBondTypeAsDouble() for b in s._rdkit.GetBonds())
    print('  %s  aromatic=%s bond_orders=%s' % (s.get_formula(), arom, orders))

merged = _merge_molecules(structures)
before = sorted(b.GetBondTypeAsDouble() for b in merged.GetBonds())
print('--- merged (before re-aromatize) bond orders:', before)
_re_aromatize(merged)
after = sorted(b.GetBondTypeAsDouble() for b in merged.GetBonds())
print('--- merged (after re-aromatize) bond orders:', after)
arom = [a.GetIdx() for a in merged.GetAtoms() if a.GetIsAromatic()]
print('    aromatic atoms after re-aromatize:', arom)
n_benzene = sum(1 for b in merged.GetBonds() if _is_benzene(b.GetBondTypeAsDouble()))
print('    # benzene(1.5) bonds:', n_benzene)

# Now apply recipe manually and see
try:
    recipe._apply(merged, True, True)
    print('--- after recipe bond orders:', sorted(b.GetBondTypeAsDouble() for b in merged.GetBonds()))
    print('    aromatic atoms:', [a.GetIdx() for a in merged.GetAtoms() if a.GetIsAromatic()])
    # try kekulize
    try:
        m2 = Chem.Mol(merged)
        Chem.Kekulize(m2, clearAromaticFlags=True)
        print('    RDKit Kekulize OK ->', sorted(b.GetBondTypeAsDouble() for b in m2.GetBonds()))
    except Exception as e:
        print('    RDKit Kekulize FAIL: %s: %s' % (type(e).__name__, e))
except Exception as e:
    print('recipe _apply EXC %s: %s' % (type(e).__name__, e))

# What does RMG's recorded product look like?
print('--- RMG recorded product adjlist ---')
for p in case['reactions'][0]['products']:
    print(p)
