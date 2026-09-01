import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.core.family import Family, DEFAULT_FAMILIES_DIR, DEFAULT_KINETICS_DB
from rmgpu.molecule.group import Group

fam = Family.from_files('Birad_recombination', DEFAULT_FAMILIES_DIR,
                        DEFAULT_KINETICS_DB)
print('reactant_num:', fam.reactant_num)
print('reactant_num_stored:', fam.reactant_num_stored)
print('fwd template:', [e.label for e in fam.forward_template])
item = fam.entries['Root'].item
print('Root item type:', type(item).__name__)
print('isinstance Group:', isinstance(item, Group))
if isinstance(item, Group):
    print('split ->', len(item.split()))
