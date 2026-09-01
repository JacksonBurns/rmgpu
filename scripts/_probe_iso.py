import json, os, sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule

BASE='/home/jackson/rmgpu/rmgpu/gates/baselines/job05/step02_products_reference.json'
d=json.load(open(BASE))

def canon(m):
    return Chem.MolToSmiles(m._rdkit, canonical=True)

for case in d['cases']:
    if not (case['family']=='intra_H_migration' and case['reactant_smiles']==['C[CH]CCC']):
        continue
    print('CASE intra_H_migration C[CH]CCC, n_reactions=',case['n_reactions'])
    for i,r in enumerate(case['reactions']):
        ps=[Molecule.from_adjacency_list(t) for t in r['products']]
        smis=[canon(p) for p in ps]
        print(f"  rxn{i} deg={r['degeneracy']} products={smis}")
    # compare products across reactions for isomorphism
    print('  --- isomorphism between rxns ---')
    for i in range(case['n_reactions']):
        for j in range(i+1,case['n_reactions']):
            pi=Molecule.from_adjacency_list(case['reactions'][i]['products'][0])
            pj=Molecule.from_adjacency_list(case['reactions'][j]['products'][0])
            mi=Chem.AddHs(Chem.Mol(pi._rdkit)); mj=Chem.AddHs(Chem.Mol(pj._rdkit))
            iso = mi.HasSubstructMatch(mj) and mj.HasSubstructMatch(mi)
            print(f"  rxn{i}.prod0 vs rxn{j}.prod0 isomorphic(RDKit)={iso}  "
                  f"smi={canon(pi)!r} / {canon(pj)!r}")
