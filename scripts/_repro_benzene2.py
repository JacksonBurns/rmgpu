#!/usr/bin/env python3
"""Probe the benzene+H apply_recipe product pieces: aromatic flags, bond
orders, and whether CalcMolFormula/SanitizeMol fail. Pin the crash."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.core.recipe import ReactionRecipe, apply_recipe, ActionError, KekulizationError
import json

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'gates', 'baselines', 'job05', 'step02_products_reference.json')

data = json.load(open(BASE))
case = [c for c in data['cases'] if c['family'] == 'R_Addition_MultipleBond'
        and c['reactant_smiles'] == ['C1=CC=CC=C1', '[H]']][0]
meta = case['family_meta']
recipe = ReactionRecipe.from_data(meta['recipe'])

# Use the first recorded application's labeled reactants
from rmgpu.core import enumeration as enum
match = enum.RecordedMatcher(case)
apps = list(match.yield_applications())
print('n applications (ok):', len(apps))
for i, structures in enumerate(apps):
    try:
        prods = apply_recipe(structures, recipe, family_label=case['family'],
                             forward=True, relabel_atoms=True,
                             own_reverse=meta['own_reverse'],
                             reverse_map=meta['reverse_map'],
                             product_num=meta.get('effective_product_num_forward'),
                             electrons=meta['electrons'])
    except (ActionError, KekulizationError) as e:
        print('app %d: EXC %s: %s' % (i, type(e).__name__, e))
        continue
    if prods is None:
        print('app %d: None' % i)
        continue
    for j, p in enumerate(prods):
        m = p._rdkit
        arom = [a.GetIdx() for a in m.GetAtoms() if a.GetIsAromatic()]
        bonds = []
        for b in m.GetBonds():
            bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()))
        try:
            f = Chem.rdMolDescriptors.CalcMolFormula(m)
        except Exception as e:
            f = 'CALC-FAIL %s' % type(e).__name__
        print('app %d prod %d: formula=%s aromatic_atoms=%s bonds=%s' % (i, j, f, arom, bonds))
        if 'FAIL' in str(f):
            # try sanitize
            try:
                mm = Chem.Mol(m)
                Chem.SanitizeMol(mm)
                print('   sanitize OK ->', Chem.rdMolDescriptors.CalcMolFormula(mm))
            except Exception as e2:
                print('   sanitize FAIL %s: %s' % (type(e2).__name__, e2))
