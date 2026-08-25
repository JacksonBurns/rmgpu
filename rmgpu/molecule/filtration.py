"""
Module for filtering molecules based on structural criteria.
Provides functions for filtering out forbidden structures from a list of molecules.
"""

from rdkit import Chem
from rmgpu.molecule.molecule import Molecule


def filter_structures(mol_list, forbidden_structures, mark_forbidden=True):
    """
    Filter a list of molecules, returning only those that do not contain any
    forbidden structures.

    Args:
        mol_list: List of Molecule objects to filter
        forbidden_structures: List of Molecule objects representing forbidden structures
        mark_forbidden: If True, add a 'forbidden' property to molecules that match

    Returns:
        List of filtered Molecule objects (those that don't match any forbidden structure)
    """
    if not mol_list:
        return []
    
    if not forbidden_structures:
        return list(mol_list)
    
    # Parse forbidden structures to queries
    forbidden_queries = []
    for forbidden in forbidden_structures:
        query = _to_query(forbidden)
        if query is not None:
            forbidden_queries.append(query)
    
    if not forbidden_queries:
        return list(mol_list)
    
    filtered_list = []
    for mol in mol_list:
        is_forbidden = False
        
        # Check if molecule matches any forbidden structure
        for query in forbidden_queries:
            if _matches(mol._rdkit, query):
                is_forbidden = True
                if mark_forbidden:
                    _mark_forbidden(mol, query)
                break
        
        if not is_forbidden:
            filtered_list.append(mol)
    
    return filtered_list


def _to_query(molecule):
    """
    Convert a Molecule to a query object.
    Uses the molecule's SMILES as a SMARTS query.
    
    Args:
        molecule: The Molecule object to convert
        
    Returns:
        An RDKit Mol object representing the query, or None if conversion fails
    """
    try:
        smiles = molecule.to_smiles()
        query = Chem.MolFromSmarts(smiles)
        return query
    except Exception:
        return None


def _matches(target_rdmol, query_rdmol):
    """
    Check if target molecule matches the query.
    
    Args:
        target_rdmol: The RDKit Mol to check
        query_rdmol: The RDKit Mol query
        
    Returns:
        bool: True if the molecule matches the query
    """
    if target_rdmol is None or query_rdmol is None:
        return False
    
    try:
        return target_rdmol.HasSubstructMatch(query_rdmol)
    except Exception:
        return False


def _mark_forbidden(molecule, query):
    """
    Mark a molecule as forbidden by adding a property.
    """
    rdmol = molecule._rdkit
    rwmol = Chem.RWMol(rdmol)
    rwmol.SetProp('forbidden', 'True')
    molecule._rdkit = Chem.Mol(rwmol)


def is_forbidden(mol, forbidden_structures):
    """
    Check if a molecule contains any forbidden structure.
    
    Args:
        mol: The Molecule to check
        forbidden_structures: List of Molecule objects representing forbidden structures
        
    Returns:
        True if the molecule contains any forbidden structure, False otherwise
    """
    if not mol or not forbidden_structures:
        return False
    
    # Parse forbidden structures to queries
    forbidden_queries = []
    for forbidden in forbidden_structures:
        query = _to_query(forbidden)
        if query is not None:
            forbidden_queries.append(query)
    
    if not forbidden_queries:
        return False
    
    # Check if molecule matches any forbidden structure
    for query in forbidden_queries:
        if _matches(mol._rdkit, query):
            return True
    
    return False
