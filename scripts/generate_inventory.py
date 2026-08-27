#!/usr/bin/env python
"""
Generate DSL inventory: 47 files x functions used.
"""
import os
import re
import json
from collections import defaultdict

# List of all 41 DSL functions from rmgpy.rmg.input
ALL_FUNCTIONS = [
    'database', 'catalystProperties', 'coreSpeciesFile', 'species', 'forbidden',
    'SMARTS', 'fragment_adj', 'fragment_SMILES', 'SMILES', 'InChI',
    'adjacencyList', 'adjacencyListGroup', 'react', 'simpleReactor',
    'constantVIdealGasReactor', 'constantTPIdealGasReactor', 'liquidSurfaceReactor',
    'constantTVLiquidReactor', 'liquidReactor', 'surfaceReactor', 'mbsampledReactor',
    'simulator', 'solvation', 'liquidVolumetricMassTransferCoefficientPowerLaw',
    'model', 'quantumMechanics', 'mlEstimator', 'pressureDependence', 'options',
    'generatedSpeciesConstraints', 'thermoCentralDatabase', 'uncertainty',
    'restartFromSeed', 'SolventData', 'species_dict', 'mol_to_frag',
    'set_global_rmg', 'read_input_file', 'read_thermo_input_file', 'save_input_file',
    'get_input'
]

# Map DSL function names to the function names in local_context
DSL_NAMES = {
    'database': 'database',
    'catalystProperties': 'catalyst_properties',
    'coreSpeciesFile': 'core_species_file',
    'species': 'species',
    'forbidden': 'forbidden',
    'SMARTS': 'smarts',
    'fragment_adj': 'fragment_adj',
    'fragment_SMILES': 'fragment_smiles',
    'SMILES': 'smiles',
    'InChI': 'inchi',
    'adjacencyList': 'adjacency_list',
    'adjacencyListGroup': 'adjacency_list_group',
    'react': 'react',
    'simpleReactor': 'simple_reactor',
    'constantVIdealGasReactor': 'constant_V_ideal_gas_reactor',
    'constantTPIdealGasReactor': 'constant_TP_ideal_gas_reactor',
    'liquidSurfaceReactor': 'liquid_cat_reactor',
    'constantTVLiquidReactor': 'constant_T_V_liquid_reactor',
    'liquidReactor': 'liquid_reactor',
    'surfaceReactor': 'surface_reactor',
    'mbsampledReactor': 'mb_sampled_reactor',
    'simulator': 'simulator',
    'solvation': 'solvation',
    'liquidVolumetricMassTransferCoefficientPowerLaw': 'liquid_volumetric_mass_transfer_coefficient_power_law',
    'model': 'model',
    'quantumMechanics': 'quantum_mechanics',
    'mlEstimator': 'ml_estimator',
    'pressureDependence': 'pressure_dependence',
    'options': 'options',
    'generatedSpeciesConstraints': 'generated_species_constraints',
    'thermoCentralDatabase': 'thermo_central_database',
    'uncertainty': 'uncertainty',
    'restartFromSeed': 'restart_from_seed',
    'SolventData': 'SolventData',
}

# Reverse map: function name in file -> DSL name
FUNCTION_TO_DSL = {v: k for k, v in DSL_NAMES.items()}

def find_input_files():
    """Find all 47 input.py files."""
    files = []
    
    # From examples/rmg/
    examples_rmg = '/home/jackson/rmgpu/RMG-Py/examples/rmg'
    for dirpath, dirnames, filenames in os.walk(examples_rmg):
        if 'input.py' in filenames:
            files.append(os.path.join(dirpath, 'input.py'))
    
    # From test/regression/
    test_regression = '/home/jackson/rmgpu/RMG-Py/test/regression'
    if os.path.exists(test_regression):
        for dirpath, dirnames, filenames in os.walk(test_regression):
            if 'input.py' in filenames:
                files.append(os.path.join(dirpath, 'input.py'))
    
    # From examples/scripts/ (but these may be different)
    # Let's skip these for now as they seem to be script-based
    
    return sorted(files)

def analyze_file(filepath):
    """Analyze a single input.py file for DSL functions used."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find all function calls using regex
    # Pattern: function_name( followed by optional whitespace
    function_pattern = r'\b(\w+)\s*\('
    matches = re.findall(function_pattern, content)
    
    # Filter to only DSL functions
    used_functions = set()
    for func in matches:
        if func in DSL_NAMES:
            used_functions.add(func)
    
    return used_functions

def main():
    files = find_input_files()
    print(f"Found {len(files)} input.py files")
    
    inventory = defaultdict(set)  # function -> set of files
    
    for filepath in files:
        # Get relative path for display
        relpath = os.path.relpath(filepath, '/home/jackson/rmgpu/RMG-Py')
        used = analyze_file(filepath)
        print(f"\n{relpath}:")
        for func in sorted(used):
            print(f"  - {func}")
            inventory[func].add(relpath)
    
    # Print summary
    print("\n" + "="*60)
    print("DSL FUNCTION SUMMARY")
    print("="*60)
    
    # Functions in ALL_FUNCTIONS that are not DSL (internal)
    internal_functions = set(ALL_FUNCTIONS) - set(DSL_NAMES.keys())
    print(f"\nInternal functions (not DSL): {len(internal_functions)}")
    
    # Sort by count
    sorted_functions = sorted(inventory.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\nDSL functions used in files: {len(sorted_functions)}")
    print("\nFunction -> Count -> Files:")
    for func, files_set in sorted_functions:
        print(f"\n{func} ({len(files_set)} files):")
        for f in sorted(files_set):
            print(f"  - {f}")

if __name__ == '__main__':
    main()