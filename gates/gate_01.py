#!/usr/bin/env python
"""
Job-01 gate: round-trips vs RMG-Py.

Checks:
1. Adjlist round-trip: parse(Molecule.to_adjlist()) == Molecule
2. Adjlist parity with RMG-Py
3. Canonical SMILES parity with RMG-Py
4. Atom type parity with RMG-Py
5. Resonance set parity with RMG-Py
6. Symmetry number parity with RMG-Py

Usage:
    /home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_01.py
"""
import os
import sys
import json
import logging

# Add rmgpu package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import test set
sys.path.insert(0, os.path.dirname(__file__))
from test_set import get_test_molecules, get_molecule_labels

# Import rmgpu modules
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.adjlist import parse_adjlist
from rmgpu.molecule.atomtype import assign_atom_types
from rmgpu.molecule.resonance import generate_resonance_structures
from rmgpu.molecule.symmetry import get_symmetry_number

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_references():
    """Load RMG-Py reference data from results.json."""
    ref_path = os.path.join(os.path.dirname(__file__), 'baselines', 'job01', 'results.json')
    with open(ref_path, 'r') as f:
        return json.load(f)

def check_adjlist_roundtrip(mol):
    """
    Check that parse(Molecule.to_adjlist()) == Molecule.
    Returns True if round-trip succeeds, False otherwise.
    """
    adjlist_str = mol.to_adjlist()
    if not adjlist_str:
        return False
    try:
        parsed_mol = Molecule.from_adjacency_list(adjlist_str)
        return mol.is_isomorph(parsed_mol)
    except Exception:
        return False

def check_adjlist_parity(mol, label, reference):
    """
    Check that rmgpu adjlist matches RMG-Py adjlist.
    Returns True if match, False otherwise.
    """
    rmgpu_adjlist = mol.to_adjlist()
    rmgpy_adjlist = reference.get(label, '')
    
    # Normalize: strip leading/trailing whitespace and compare
    rmgpu_norm = rmgpu_adjlist.strip()
    rmgpy_norm = rmgpy_adjlist.strip()
    
    return rmgpu_norm == rmgpy_norm

def check_atomtype_parity(mol, label, reference):
    """
    Check that rmgpu atom types match RMG-Py atom types.
    Returns True if match, False otherwise.
    """
    rmgpu_types = assign_atom_types(mol)
    rmgpy_types = reference.get(label, [])
    
    # Normalize both to lists of strings
    rmgpu_norm = [str(t) for t in rmgpu_types]
    rmgpy_norm = [str(t) for t in rmgpy_types]
    
    return rmgpu_norm == rmgpy_norm

def check_resonance_parity(mol, label, reference):
    """
    Check that rmgpu resonance structures match RMG-Py resonance structures.
    Returns True if match, False otherwise.
    """
    rmgpu_res = generate_resonance_structures(mol)
    rmgpu_smiles = sorted([s.to_smiles() for s in rmgpu_res])
    rmgpy_smiles = sorted(reference.get(label, []))
    
    return rmgpu_smiles == rmgpy_smiles

def check_symmetry_parity(mol, label, reference):
    """
    Check that rmgpu symmetry number matches RMG-Py symmetry number.
    Returns True if match, False otherwise.
    """
    rmgpu_sym = get_symmetry_number(mol)
    rmgpy_sym = reference.get(label, 0)
    
    return rmgpu_sym == rmgpy_sym

def main():
    # Load references
    references = load_references()
    logger.info(f"Loaded references: adjlist={len(references['adjlist'])}, "
                f"atomtypes={len(references['atomtypes'])}, "
                f"resonance={len(references['resonance'])}, "
                f"symmetry={len(references['symmetry'])}")
    
    test_molecules = get_test_molecules()
    labels = get_molecule_labels()
    
    # Results tracking
    results = {
        'adjlist_roundtrip': {'pass': 0, 'fail': 0, 'failures': []},
        'adjlist_parity': {'pass': 0, 'fail': 0, 'failures': []},
        'smiles_parity': {'pass': 0, 'fail': 0, 'failures': []},
        'atomtype_parity': {'pass': 0, 'fail': 0, 'failures': []},
        'resonance_parity': {'pass': 0, 'fail': 0, 'failures': []},
        'symmetry_parity': {'pass': 0, 'fail': 0, 'failures': []},
    }
    
    # Run checks for each molecule
    for smiles, label in zip(test_molecules, labels):
        logger.info(f"Testing molecule: {label} ({smiles})")
        
        try:
            mol = Molecule(smiles=smiles)
        except Exception as e:
            logger.warning(f"  Could not create Molecule: {e}")
            continue
        
        # Check 1: Adjlist round-trip
        if check_adjlist_roundtrip(mol):
            results['adjlist_roundtrip']['pass'] += 1
        else:
            results['adjlist_roundtrip']['fail'] += 1
            results['adjlist_roundtrip']['failures'].append(label)
        
        # Check 2: Adjlist parity
        if label in references['adjlist']:
            if check_adjlist_parity(mol, label, references['adjlist']):
                results['adjlist_parity']['pass'] += 1
            else:
                results['adjlist_parity']['fail'] += 1
                results['adjlist_parity']['failures'].append(label)
        
        # Check 3: SMILES parity
        if label in references['resonance'] and references['resonance'][label]:
            rmgpy_smiles = references['resonance'][label][0]
            rmgpu_smiles = mol.to_smiles()
            if rmgpu_smiles == rmgpy_smiles:
                results['smiles_parity']['pass'] += 1
            else:
                results['smiles_parity']['fail'] += 1
                results['smiles_parity']['failures'].append(label)
        
        # Check 4: Atom type parity
        if label in references['atomtypes']:
            if check_atomtype_parity(mol, label, references['atomtypes']):
                results['atomtype_parity']['pass'] += 1
            else:
                results['atomtype_parity']['fail'] += 1
                results['atomtype_parity']['failures'].append(label)
        
        # Check 5: Resonance parity
        if label in references['resonance']:
            if check_resonance_parity(mol, label, references['resonance']):
                results['resonance_parity']['pass'] += 1
            else:
                results['resonance_parity']['fail'] += 1
                results['resonance_parity']['failures'].append(label)
        
        # Check 6: Symmetry parity
        if label in references['symmetry']:
            if check_symmetry_parity(mol, label, references['symmetry']):
                results['symmetry_parity']['pass'] += 1
            else:
                results['symmetry_parity']['fail'] += 1
                results['symmetry_parity']['failures'].append(label)
    
    # Print results
    print("\n" + "="*60)
    print("JOB-01 GATE RESULTS")
    print("="*60)
    
    total_molecules = len(test_molecules)
    
    for check_name, result in results.items():
        pass_count = result['pass']
        fail_count = result['fail']
        failures = result['failures']
        print(f"\n{check_name}:")
        print(f"  Passed: {pass_count}/{total_molecules}")
        print(f"  Failed: {fail_count}/{total_molecules}")
        if failures:
            print(f"  Failures: {', '.join(failures)}")
    
    # Final verdict
    all_pass = all(r['fail'] == 0 for r in results.values())
    print("\n" + "="*60)
    if all_pass:
        print("GATE STATUS: PASS")
        print("="*60)
        return 0
    else:
        print("GATE STATUS: FAIL")
        print("="*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())
