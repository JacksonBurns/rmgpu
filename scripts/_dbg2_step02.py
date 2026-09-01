import json, os, sys, traceback
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import enumeration as enum

BASE='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step02_products_reference.json'
case=json.load(open(BASE))['cases'][0]
fam=enum.Family.from_reference(case)
reactants=[Molecule(smiles=s) for s in case['reactant_smiles']]
fam.matcher=enum.RecordedMatcher(case)

# trace the replay path
rxn_list=[]
napps=0
for structures in fam.matcher.yield_applications():
    napps+=1
    if len(structures)!=fam.reactant_num_effective:
        print('skip (len)')
        continue
    from rmgpu.core.recipe import apply_recipe, ActionError, KekulizationError
    try:
        prods=apply_recipe(structures, fam.recipe, family_label=fam.label, forward=True,
                           relabel_atoms=True, own_reverse=fam.own_reverse,
                           reverse_map=fam.reverse_map, product_num=fam.product_num_forward,
                           electrons=fam.electrons)
    except (ActionError,KekulizationError) as e:
        print('app',napps,'EXC',type(e).__name__,e)
        continue
    print('app',napps,'products:',prods)
    if not prods:
        continue
    rxn=enum.create_reaction(fam, structures, prods, True)
    print('app',napps,'create_reaction ->',rxn)
    if rxn is not None:
        rxn_list.append(rxn)
print('napps:',napps,'rxn_list:',len(rxn_list))
collapsed=enum.find_degenerate_reactions(rxn_list, same_reactants=0, template=None, family=fam, resonance=True)
print('collapsed:',collapsed)
