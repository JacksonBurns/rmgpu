"""
Test set for job-01 gate.

Contains molecules from superminimal, c3h4, and minimal examples.
"""

# SMILES strings for test molecules
TEST_MOLECULES = [
    # From superminimal (use explicit hydrogens)
    "[H][H]",      # H2
    "[O][O]",      # O2
    
    # From c3h4
    "[CH2]",       # CH2 (carbene)
    "C#C",         # C2H2 (acetylene)
    "[N][N]",      # N2
    
    # From minimal
    "CC",          # ethane
    
    # Additional common molecules for robustness
    "C",           # methane
    "O",           # water
    "C=C",         # ethylene
    "CC(C)C",      # isobutane
    "[CH3]",       # methyl radical
    "CO",          # formaldehyde
    "CC=O",        # acetaldehyde
    "O=C=O",       # carbon dioxide
    "c1ccccc1",    # benzene
    "c1ccccc1C",   # toluene
    "[N+](=O)[O-]", # NO2 (nitrogen dioxide)
    "[O-][N+](=O)", # HNO3-like
    "OCC=O",       # glycolaldehyde
]

# Molecule labels for reporting (must match TEST_MOLECULES order)
MOLECULE_LABELS = [
    "H2", "O2", "CH2", "C2H2", "N2",
    "ethane", "methane", "water", "ethylene", "isobutane",
    "methyl_radical", "formaldehyde", "acetaldehyde", "CO2",
    "benzene", "toluene", "NO2", "HNO3-like", "glycolaldehyde"
]

def get_test_molecules():
    """Return list of test molecule SMILES strings."""
    return TEST_MOLECULES

def get_molecule_labels():
    """Return list of molecule labels for reporting."""
    return MOLECULE_LABELS
