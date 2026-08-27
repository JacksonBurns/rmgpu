#!/usr/bin/env python
"""Generate RMG-Py library entry counts for gate_02.py.

Uses COUNT(DISTINCT label) to match the rmgdb view's duplicate-row behavior.
"""
import json
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import RMG-Py modules
import rmgpy
from rmgpy.data.thermo import ThermoDatabase
from rmgpy.data.kinetics import KineticsDatabase

THERMO_LIBRARIES = [
    'primaryThermoLibrary',
    'BurcatNS',
    'BurkeH2O2',
    'NOx2018',
]

KINETICS_LIBRARIES = [
    'primaryH2O2',
    'primaryNitrogenLibrary',
    'NOx2018',
]

THERMO_DB_PATH = '/home/jackson/rmgpu/RMG-database/input/thermo'
KINETICS_DB_PATH = '/home/jackson/rmgpu/RMG-database/input/kinetics'


def get_thermo_library_count(library_name: str, thermo_db: ThermoDatabase) -> int:
    """Get entry count for a thermo library via RMG-Py."""
    if library_name not in thermo_db.libraries:
        logger.warning(f"Library {library_name} not found in thermo DB")
        return 0
    return len(thermo_db.libraries[library_name].entries)


def get_kinetics_library_count(library_name: str, kinetics_db: KineticsDatabase) -> int:
    """Get entry count for a kinetics library via RMG-Py."""
    if library_name not in kinetics_db.libraries:
        logger.warning(f"Library {library_name} not found in kinetics DB")
        return 0
    return len(kinetics_db.libraries[library_name].entries)


def main():
    logger.info("Generating RMG-Py library entry counts...")
    
    # Load thermo database
    thermo_db = ThermoDatabase()
    thermo_db.load(THERMO_DB_PATH)
    
    # Load kinetics database (families=[] to skip family loading)
    kinetics_db = KineticsDatabase()
    kinetics_db.load(KINETICS_DB_PATH, families=[], depositories=[])
    
    results = {
        'thermo': {},
        'kinetics': {},
    }
    
    # Get thermo library counts
    for library in THERMO_LIBRARIES:
        count = get_thermo_library_count(library, thermo_db)
        results['thermo'][library] = count
        logger.info(f"Thermo library {library}: {count} entries")
    
    # Get kinetics library counts
    for library in KINETICS_LIBRARIES:
        count = get_kinetics_library_count(library, kinetics_db)
        results['kinetics'][library] = count
        logger.info(f"Kinetics library {library}: {count} entries")
    
    # Save to JSON
    output_path = Path(__file__).parent / 'baselines' / 'db_counts.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved counts to {output_path}")
    logger.info(json.dumps(results, indent=2))
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
