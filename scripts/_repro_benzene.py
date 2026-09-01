#!/usr/bin/env python3
"""Repro the benzene+H AtomValenceException in generate_reactions."""
import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import enumeration as enum
import json

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'gates', 'baselines', 'job05', 'step02_products_reference.json')

data = json.load(open(BASE))
case = [c for c in data['cases'] if c['family'] == 'R_Addition_MultipleBond'
        and c['reactant_smiles'] == ['C1=CC=CC=C1', '[H]']][0]

fam = enum.Family.from_reference(case)
print('reactant_smiles', case['reactant_smiles'])
# Step 1: resonance expansion
for s in case['reactant_smiles']:
    m = Molecule(smiles=s)
    print('building', s, '->', m.to_smiles())
    from rmgpu.molecule.resonance import generate_resonance_structures
    try:
        forms = generate_resonance_structures(m)
        print('  resonance %d forms' % len(forms))
    except Exception:
        traceback.print_exc()
# Step 2: full generate_reactions
fam.matcher = enum.RecordedMatcher(case)
reactants = [Molecule(smiles=s) for s in case['reactant_smiles']]
try:
    rxns = enum.generate_reactions(fam, reactants)
    print('OK generate_reactions ->', len(rxns), 'reactions')
except Exception:
    traceback.print_exc()
