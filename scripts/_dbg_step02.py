import json, os, sys, traceback
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import enumeration as enum
from rmgpu.core.recipe import apply_recipe, ActionError, KekulizationError

BASE='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step02_products_reference.json'
case=json.load(open(BASE))['cases'][0]  # H_Abstraction C + [H]
fam=enum.Family.from_reference(case)
print('family label:',fam.label,'own_reverse:',fam.own_reverse,'reactant_num_eff:',fam.reactant_num_effective,'prod_num_fwd:',fam.product_num_forward)
print('reverse_map:',fam.reverse_map)
app=case['applications'][0]
print('app ok:',app.get('ok'))
print('labeled_reactants:')
for i,t in enumerate(app['labeled_reactants']):
    print('--- struct',i,'---')
    print(t)
# build structures like RecordedMatcher does
structures=[Molecule.from_adjacency_list(t) for t in app['labeled_reactants']]
for struct,idlist in zip(structures,app['reactant_atom_ids']):
    for atom,aid in zip(struct._rdkit.GetAtoms(),idlist):
        atom.SetProp('atomid',str(int(aid)))
for s in structures:
    labs=[(a.GetProp('label') if a.HasProp('label') else '', a.GetProp('atomid') if a.HasProp('atomid') else '') for a in s._rdkit.GetAtoms()]
    print('struct labels/ids:',labs)
# try apply_recipe
try:
    prods=apply_recipe(structures, fam.recipe, family_label=fam.label, forward=True,
                       relabel_atoms=True, own_reverse=fam.own_reverse, reverse_map=fam.reverse_map,
                       product_num=fam.product_num_forward, electrons=fam.electrons)
    print('apply_recipe returned:',prods)
except (ActionError,KekulizationError) as e:
    print('EXC',type(e).__name__,e)
    traceback.print_exc()
