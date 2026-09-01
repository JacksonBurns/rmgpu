import json, sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.atomtype import assign_atom_types
from rmgpu.molecule import group as G
AT = G.atom_type
m = Molecule(smiles='C[CH]CCC')
types = assign_atom_types(m)
print("types:", types)
# how many radical atoms
em = m._with_explicit_h()
for a in em.GetAtoms():
    print(f"  idx{a.GetIdx()} {a.GetSymbol()} rad={a.GetNumRadicalElectrons()} charge={a.GetFormalCharge()} nBonds={a.GetTotalNumHs()+len(a.GetBonds())} type={types[a.GetIdx()]}")
# atom type check for R!H
gat = G.atom_type('R!H')
print("R!H generic:", [g.label for g in gat.generic])
print("R!H specific (count):", len(gat.specific))
print("Cs is_specific_case_of R!H:", AT('Cs').is_specific_case_of(gat) if AT('Cs') else "Cs NOT a type")
print("C is_specific_case_of R!H:", AT('C').is_specific_case_of(gat) if AT('C') else "C NOT a type")
print("R!H is_specific_case_of R!H:", gat.is_specific_case_of(gat))
print("atom_type('Cs') is:", AT('Cs'))
