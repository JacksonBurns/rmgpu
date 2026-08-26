#!/usr/bin/env python3
"""Focused probe: how RMG-Py builds molecules from these SMILES."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/RMG-Py')
from rmgpy.molecule.molecule import Molecule as RMGMolecule
from rdkit import Chem

for smi in ['CO', 'CC=O', 'O=C=O', 'c1ccccc1', 'c1ccccc1C', '[N+](=O)[O-]', '[O-][N+](=O)', 'OCC=O', '[H][H]', '[CH2]']:
    rd = Chem.MolFromSmiles(smi)
    rd_h = Chem.AddHs(rd)
    print(f"=== {smi}")
    print(f"  RDKit: n_atoms={rd.GetNumAtoms()}, withH={rd_h.GetNumAtoms()}, bonds={[(b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol(), str(b.GetBondType())) for b in rd.GetBonds()]}")
    try:
        mol = RMGMolecule(smiles=smi)
        print(f"  RMG: n_vertices={len(mol.vertices)}, to_smiles={mol.to_smiles()}, mult={mol.multiplicity}")
        print(f"  RMG bonds: {[(mol.vertices[i].symbol, mol.vertices[j].symbol, b.order) for i, v in enumerate(mol.vertices) for j, b in v.bonds.items() if i < j]}")
        print(f"  adjlist: {mol.to_adjacency_list()!r}")
    except Exception as e:
        print(f"  RMG ERROR: {type(e).__name__}: {e}")
