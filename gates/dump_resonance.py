#!/usr/bin/env python
"""
Reference dump script for resonance structures using RMG-Py.

This script generates resonance structures using RMG-Py's Molecule.get_resonance_structures()
and outputs them to the gates/baselines/resonance/ directory.

Usage:
    python dump_resonance.py
"""

import os
import sys
from pathlib import Path

# Add RMG-Py to the path
sys.path.insert(0, '/home/jackson/rmgpu/RMG-Py')

from rmgpy.molecule.molecule import Molecule
from rmgpy.molecule.resonance import generate_resonance_structures


# Test molecules (same as in the test file)
TEST_MOLECULES = [
    ("[CH3]", "methyl_radical"),
    ("O=C=O", "co2"),
    ("[OH-]", "hydroxide_ion"),
    ("[NH4+]", "ammonium_ion"),
    ("c1ccccc1", "benzene"),
    ("c1ccc2ccccc2c1", "naphthalene"),
    ("C=CC=C", "allene")
]


def main():
    # Create output directory
    output_dir = Path("/home/jackson/rmgpu/rmgpu/gates/baselines/resonance")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating resonance structures for {len(TEST_MOLECULES)} molecules...")
    
    for smiles, name in TEST_MOLECULES:
        try:
            # Create the molecule
            mol = Molecule(smiles=smiles)
            
            # Generate resonance structures
            resonance_structures = generate_resonance_structures(mol)
            
            # Convert to SMILES set
            smiles_set = {rs.to_smiles() for rs in resonance_structures}
            
            # Write to output file
            output_file = output_dir / f"{name}.txt"
            with open(output_file, 'w') as f:
                f.write(f"# Molecule: {smiles}\n")
                f.write(f"# Resonance structures: {len(smiles_set)}\n\n")
                for smi in sorted(smiles_set):
                    f.write(f"{smi}\n")
            
            print(f"  {name}: {len(smiles_set)} resonance structures")
            
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    print("Done!")


if __name__ == "__main__":
    main()
