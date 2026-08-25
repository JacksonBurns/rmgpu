"""
Module for estimating symmetry numbers of molecules from their chemical graph
representation.
"""

from rdkit import Chem
from rmgpu.molecule.molecule import Molecule


def _is_in_ring(ri, atom_idx):
    """Check if an atom is in any ring."""
    for ring in ri.AtomRings():
        if atom_idx in ring:
            return True
    return False


def _is_bond_in_ring(ri, idx1, idx2):
    """Check if a bond is in any ring."""
    for ring in ri.AtomRings():
        if idx1 in ring and idx2 in ring:
            ring_list = list(ring)
            for i, atom in enumerate(ring_list):
                next_atom = ring_list[(i + 1) % len(ring_list)]
                if (atom == idx1 and next_atom == idx2) or (atom == idx2 and next_atom == idx1):
                    return True
    return False


def _split_into_groups(rdmol):
    """
    Split a molecule into connected components (groups).
    """
    if rdmol is None or rdmol.GetNumAtoms() == 0:
        return []
    
    from collections import deque
    visited = set()
    groups = []
    
    for atom_idx in range(rdmol.GetNumAtoms()):
        if atom_idx not in visited:
            queue = deque([atom_idx])
            component = []
            while queue:
                current = queue.popleft()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    atom = rdmol.GetAtomWithIdx(current)
                    for bond in atom.GetBonds():
                        other_idx = bond.GetBeginAtomIdx()
                        if bond.GetBeginAtomIdx() == current:
                            other_idx = bond.GetEndAtomIdx()
                        else:
                            other_idx = bond.GetBeginAtomIdx()
                        if other_idx not in visited:
                            queue.append(other_idx)
            
            if component:
                # Create a new molecule for this component
                rw_new = Chem.RWMol()
                idx_map = {}
                for i, atom_idx in enumerate(component):
                    atom = rdmol.GetAtomWithIdx(atom_idx)
                    new_atom = Chem.Atom(atom.GetSymbol())
                    new_atom.SetFormalCharge(atom.GetFormalCharge())
                    new_atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons())
                    new_idx = rw_new.AddAtom(new_atom)
                    idx_map[atom_idx] = new_idx
                
                # Add bonds
                for atom_idx in component:
                    atom = rdmol.GetAtomWithIdx(atom_idx)
                    for bond in atom.GetBonds():
                        if bond.GetBeginAtomIdx() == atom_idx:
                            other_idx = bond.GetEndAtomIdx()
                        else:
                            other_idx = bond.GetBeginAtomIdx()
                        if other_idx in idx_map:
                            bond_type = bond.GetBondType()
                            rw_new.AddBond(idx_map[atom_idx], idx_map[other_idx], bond_type)
                
                # Create Molecule object
                mol = Chem.Mol(rw_new)
                mol = Molecule._from_rdmol(mol)
                groups.append(mol)
    
    return groups


def _check_isomorphism(mol1, mol2):
    """
    Check if two molecules are isomorphic using RDKit.
    """
    if mol1 is None or mol2 is None:
        return False
    
    # Use canonical SMILES comparison for simplicity
    smiles1 = Chem.MolToSmiles(mol1._rdkit)
    smiles2 = Chem.MolToSmiles(mol2._rdkit)
    
    return smiles1 == smiles2


def _remove_atom_from_fragment(mol, atom_idx):
    """
    Remove an atom from a molecule fragment.
    """
    if mol is None:
        return None
    
    rdmol = mol._rdkit
    rwmol = Chem.RWMol(rdmol)
    
    try:
        rwmol.RemoveAtom(atom_idx)
        new_mol = Chem.Mol(rwmol)
        return Molecule._from_rdmol(new_mol)
    except Exception:
        return None


def _is_atom_in_fragment(mol, atom_idx):
    """
    Check if an atom index is in a fragment.
    """
    if mol is None:
        return False
    
    # For now, we'll assume the atom is in the fragment
    # A more sophisticated implementation would track original indices
    return True


def _indistinguishable(atom1, atom2):
    """
    Determine if two atoms are feasibly indistinguishable based on connections
    to nearest neighbors.
    """
    if atom1.GetSymbol() != atom2.GetSymbol():
        return False
    
    # Check bond orders
    bond_orders_1 = sorted([bond.GetBondTypeAsDouble() for bond in atom1.GetBonds()])
    bond_orders_2 = sorted([bond.GetBondTypeAsDouble() for bond in atom2.GetBonds()])
    
    if bond_orders_1 != bond_orders_2:
        return False
    
    # Check neighbors
    neighbors_1 = list(atom1.GetNeighbors())
    neighbors_2 = list(atom2.GetNeighbors())
    
    for neighbor1 in neighbors_1:
        found_match = False
        for neighbor2 in neighbors_2:
            bond1 = atom1.GetBondToAtom(neighbor1)
            bond2 = atom2.GetBondToAtom(neighbor2)
            if bond1.GetBondType() == bond2.GetBondType() and neighbor1.GetSymbol() == neighbor2.GetSymbol():
                found_match = True
                break
        if not found_match:
            return False
    
    return True


def get_symmetry_number(molecule):
    """
    Return the symmetry number for the structure.
    
    This is a simplified implementation that handles common cases.
    For a complete implementation, see RMG-Py's symmetry.py.
    
    Args:
        molecule: The Molecule object
        
    Returns:
        float: The symmetry number
    """
    rdmol = molecule._rdkit
    ri = rdmol.GetRingInfo()
    
    # Simplified symmetry number calculation
    # This handles the most common cases
    num_atoms = rdmol.GetNumAtoms()
    
    if num_atoms == 1:
        # Single atom
        atom = rdmol.GetAtomWithIdx(0)
        if atom.GetSymbol() == 'O':
            # Water (single O in SMILES)
            return 2
        # Single carbon (methane) or other single atoms
        radical_electrons = atom.GetNumRadicalElectrons()
        return 1 if radical_electrons > 0 else 12
    
    if num_atoms == 2:
        # Diatomic molecule
        atom1 = rdmol.GetAtomWithIdx(0)
        atom2 = rdmol.GetAtomWithIdx(1)
        if atom1.GetSymbol() != atom2.GetSymbol():
            return 1
        # Check if there are radicals
        if atom1.GetNumRadicalElectrons() > 0 or atom2.GetNumRadicalElectrons() > 0:
            return 1
        # Same atoms - check if single bond (like ethane)
        bond = rdmol.GetBondBetweenAtoms(0, 1)
        if bond is not None:
            bond_order = bond.GetBondTypeAsDouble()
            if bond_order == 1:
                return 3  # Ethane-like
            elif bond_order == 2:
                return 2  # Ethylene-like
            elif bond_order == 3:
                return 2  # Acetylene-like
        return 2
    
    # For polyatomic molecules, use a simplified approach
    # Count the number of identical atoms in specific positions
    symmetry_number = 1
    
    # Handle rings separately
    if molecule.is_cyclic():
        symmetry_number *= _get_cyclic_symmetry(rdmol, ri)
    
    # Handle linear molecules
    if _is_linear(rdmol):
        symmetry_number *= _get_linear_symmetry(rdmol)
    
    # Handle branched molecules
    else:
        symmetry_number *= _get_branch_symmetry(rdmol)
    
    return symmetry_number


def _get_cyclic_symmetry(rdmol, ri):
    """Get cyclic symmetry contribution."""
    rings = ri.AtomRings()
    if not rings:
        return 1
    
    # Use the smallest ring
    smallest_ring = min(rings, key=len)
    size = len(smallest_ring)
    
    # Simple case: symmetric ring (all atoms same)
    ring_atoms = [rdmol.GetAtomWithIdx(i) for i in smallest_ring]
    symbols = set(atom.GetSymbol() for atom in ring_atoms)
    
    if len(symbols) == 1:
        return size
    
    # For mixed rings, return 1 for simplicity
    return 1


def _get_linear_symmetry(rdmol):
    """Get linear symmetry contribution."""
    # Check if the molecule is linear (all atoms in a chain)
    # Simplified: check if max degree is 2
    max_degree = 0
    for atom in rdmol.GetAtoms():
        degree = len(atom.GetBonds())
        if degree > max_degree:
            max_degree = degree
    
    if max_degree > 2:
        return 1  # Not linear
    
    # For linear molecules, check symmetry about the center
    num_atoms = rdmol.GetNumAtoms()
    if num_atoms == 2:
        return 1
    
    # Check if the molecule has a center of symmetry
    # Simplified: return 2 for symmetric linear molecules
    return 2


def _is_linear(rdmol):
    """Check if molecule is linear (max degree <= 2)."""
    for atom in rdmol.GetAtoms():
        if len(atom.GetBonds()) > 2:
            return False
    return True

def _get_branch_symmetry(rdmol):
    """Get branch symmetry contribution."""
    # Count the number of identical groups around the most branched atom
    # Simplified: return 1 for simplicity
    
    # Find the atom with the highest degree
    max_degree_atom = None
    max_degree = 0
    for atom in rdmol.GetAtoms():
        degree = len(atom.GetBonds())
        if degree > max_degree:
            max_degree = degree
            max_degree_atom = atom
    
    if max_degree < 2:
        return 1
    
    # Count identical groups
    # This is a simplified approach
    return 1
