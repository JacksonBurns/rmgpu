#!/usr/bin/env python3
"""Direct ordering test: RMG-Py vertex order vs RDKit AddHs order, bond-for-bond."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/RMG-Py')
from rmgpy.molecule.molecule import Molecule as RMGMolecule
from rdkit import Chem

def rdkit_with_h(smi):
    m = Chem.MolFromSmiles(smi)
    m = Chem.AddHs(m)
    return m

for smi in ['CC=O','OCC=O','c1ccccc1C','O=C=O','CO','[N+](=O)[O-]']:
    rmg = RMGMolecule(smiles=smi)
    rmg_syms = [v.symbol for v in rmg.vertices]
    rmg_lp = [(v.symbol, v.lone_pairs, v.radical_electrons, v.charge) for v in rmg.vertices]
    rd = rdkit_with_h(smi)
    rd_syms = [a.GetSymbol() for a in rd.GetAtoms()]
    rd_lp = [(a.GetSymbol(), None, a.GetNumRadicalElectrons(), a.GetFormalCharge()) for a in rd.GetAtoms()]
    print(f"=== {smi}")
    print(f"  RMG syms : {rmg_syms}")
    print(f"  RD  syms : {rd_syms}")
    print(f"  RMG (sym,lp,rad,ch): {rmg_lp}")
    print(f"  adjlist:\n{rmg.to_adjacency_list()}")
