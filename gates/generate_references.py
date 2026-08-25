#!/usr/bin/env python
"""
Generate RMG-Py references for job-01 gate.

Dumps adjlist strings, atom types, resonance SMILES sets, and symmetry numbers
for the test set into gates/baselines/job01/.

Usage:
    /home/jackson/miniforge3/envs/rmg_env/bin/python gates/generate_references.py
"""
import os
import sys
import json
import logging

# Add RMG-Py to path
sys.path.insert(0, '/home/jackson/rmgpu/RMG-Py')

# Import test set
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu/gates')
from test_set import get_test_molecules, get_molecule_labels

# RMG-Py imports
from rmgpy.molecule.molecule import Molecule as RMGMolecule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    test_molecules = get_test_molecules()
    labels = get_molecule_labels()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'baselines', 'job01')
    adjlist_dir = os.path.join(output_dir, 'adjlist')
    atomtype_dir = os.path.join(output_dir, 'atomtypes')
    resonance_dir = os.path.join(output_dir, 'resonance')
    symmetry_dir = os.path.join(output_dir, 'symmetry')
    
    for d in [adjlist_dir, atomtype_dir, resonance_dir, symmetry_dir]:
        os.makedirs(d, exist_ok=True)
    
    results = {
        'adjlist': {},
        'atomtypes': {},
        'resonance': {},
        'symmetry': {},
        'errors': {}
    }
    
    for i, (smiles, label) in enumerate(zip(test_molecules, labels)):
        logger.info(f"Processing molecule {i+1}/{len(test_molecules)}: {label} ({smiles})")
        
        try:
            # Create RMG-Py molecule
            mol = RMGMolecule(smiles=smiles)
            
            # 1. Adjlist
            adjlist_str = mol.to_adjacency_list()
            results['adjlist'][label] = adjlist_str
            
            # 2. Atom types - in RMG-Py, atom types are assigned during initialization
            # We just need to read them from the vertices
            atom_types = [v.atomtype.label for v in mol.vertices]
            results['atomtypes'][label] = atom_types
            
            # 3. Resonance structures
            res_structures = mol.generate_resonance_structures()
            res_smiles = [s.to_smiles() for s in res_structures]
            results['resonance'][label] = res_smiles
            
            # 4. Symmetry number
            sym_num = mol.get_symmetry_number()
            results['symmetry'][label] = sym_num
            
            logger.info(f"  Success: {label}")
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            results['errors'][label] = error_msg
            logger.warning(f"  Error: {error_msg}")
    
    # Save results
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"References saved to {results_path}")
    logger.info(f"Errors: {len(results['errors'])}")
    for label, err in results['errors'].items():
        logger.warning(f"  {label}: {err}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
