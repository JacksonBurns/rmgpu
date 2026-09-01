import json, os, sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import enumeration as enum
from rmgpu.core.recipe import apply_recipe, ActionError, KekulizationError

BASE='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step02_products_reference.json'
d=json.load(open(BASE))
case=[c for c in d['cases'] if c['family']=='Intra_ene_reaction'][0]
fam=enum.Family.from_reference(case)
m=enum.RecordedMatcher(case)
print('n apps recorded:', len(case['applications']), 'ok apps:',
      sum(1 for a in case['applications'] if a.get('ok', True)))
for i, structures in enumerate(m.yield_applications()):
    try:
        prods = apply_recipe(structures, fam.recipe, family_label=fam.label,
                             forward=True, relabel_atoms=True, own_reverse=fam.own_reverse,
                             reverse_map=fam.reverse_map, product_num=fam.product_num_forward,
                             electrons=fam.electrons)
    except (ActionError, KekulizationError) as e:
        print('app%d EXC %s' % (i, e)); continue
    if not prods:
        print('app%d -> None' % i); continue
    ids = [enum.piece_atom_ids(p) for p in prods]
    from rdkit import Chem
    smis = [Chem.MolToSmiles(p._rdkit, canonical=True) for p in prods]
    print('app%d ids=%s smi=%s' % (i, ids, smis))
# raw reaction counts
rxns = enum.generate_reactions(fam, [Molecule(smiles=s) for s in case['reactant_smiles']])
print('collapsed:', [(r.degeneracy) for r in rxns])
