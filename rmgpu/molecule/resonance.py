"""Resonance structure generation for rmgpu, implementing RMG-style rules."""

from rdkit import Chem
from rdkit.Chem import BondType
from rmgpu.molecule.molecule import Molecule


def _bond_order_str(order):
    """Map bond order numeric value to a string for pattern matching."""
    if order == 1.5:
        return 'A'
    return {1: 'S', 2: 'D', 3: 'T'}.get(order, 'S')


def _is_radical(mol):
    """Return True if molecule has radical electrons."""
    return any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms())


def _get_lone_pairs(atom):
    """Estimate lone pairs from formal charge and valence."""
    # Approximate: for O/N etc., lone pairs ~ 2 - charge (positive)
    symbol = atom.GetSymbol()
    charge = atom.GetFormalCharge()
    if symbol in 'ONFS':
        return max(0, 2 - charge)
    elif symbol in 'HILC':
        return max(0, 3 - charge)
    elif symbol == 'P':
        return max(0, 3 - charge)
    elif symbol == 'Cl':
        return max(0, 3 - charge)
    return 0


def _has_nitrogen_val5(mol):
    """Check for nitrogen valence 5 atoms without lone pairs."""
    return any(atom.GetSymbol() == 'N' and atom.GetFormalCharge() >= 1 for atom in mol.GetAtoms())


def _has_lone_pairs(mol):
    """Check if molecule has any atoms with lone pairs."""
    return any(_get_lone_pairs(atom) > 0 for atom in mol.GetAtoms())


def _is_aromatic(mol):
    """Check if molecule is aromatic by detecting aromatic bonds in rings."""
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            return True
    return False


def _is_cyclic(mol):
    """Check if molecule has a ring."""
    return any(len(ring) >= 3 for ring in mol.GetRingInfo().AtomRings())


def _analyze_molecule(mol):
    """Analyze molecule features for resonance generation."""
    return {
        'is_radical': _is_radical(mol),
        'is_cyclic': _is_cyclic(mol),
        'is_aromatic': _is_aromatic(mol),
        'isPolycyclicAromatic': False,  # Simplified for now
        'is_aryl_radical': False,       # Simplified for now
        'hasNitrogenVal5': _has_nitrogen_val5(mol),
        'hasLonePairs': _has_lone_pairs(mol),
        'is_multidentate': False        # Simplified for now
    }


def _generate_allyl_delocalization_resonance_structures(mol):
    """Generate resonance structures by allyl radical shift."""
    structures = []
    if not _is_radical(mol):
        return structures
        
    # Find allyl patterns: radical adjacent to double bond
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() == 0:
            continue
            
        rad_idx = atom.GetIdx()
        
        # For each double bond in the molecule, check if radical is adjacent
        for bond in mol.GetBonds():
            if bond.GetBondType() != BondType.DOUBLE:
                continue
                
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            
            # Check if radical atom is directly adjacent to double bond
            is_adjacent_a1 = False
            is_adjacent_a2 = False
            
            for bond2 in atom.GetBonds():
                other_idx = bond2.GetOtherAtomIdx(rad_idx)
                if other_idx == a1:
                    is_adjacent_a1 = True
                elif other_idx == a2:
                    is_adjacent_a2 = True
            
            if not (is_adjacent_a1 or is_adjacent_a2):
                continue
                
            # Create new structure with shifted radical and bonds
            new_mol = Chem.RWMol(mol)
            
            # Determine target (opposite end of double bond from radical)
            if is_adjacent_a1:
                target_idx = a2
            else:
                target_idx = a1
                
            # Move radical to target
            new_mol.GetAtomWithIdx(target_idx).SetNumRadicalElectrons(
                new_mol.GetAtomWithIdx(target_idx).GetNumRadicalElectrons() + 1)
            new_mol.GetAtomWithIdx(rad_idx).SetNumRadicalElectrons(0)
            
            # Shift bond orders: double bond becomes single
            new_mol.GetBondBetweenAtoms(a1, a2).SetBondType(BondType.SINGLE)
            
            # Validate the structure
            if Chem.Kekulize(new_mol):
                structures.append(Molecule._from_rdmol(Chem.Mol(new_mol)))
    return structures


def _generate_lone_pair_multiple_bond_resonance_structures(mol):
    """Generate resonance structures by lone pair shift with double/triple bond."""
    structures = []
    if not _has_lone_pairs(mol):
        return structures
        
    # Find 3-atom systems where lone pair can shift to multiple bond
    for atom in mol.GetAtoms():
        if _get_lone_pairs(atom) == 0:
            continue
            
        # Look for adjacent atoms with multiple bonds
        for bond in atom.GetBonds():
            if bond.GetBondType() in [BondType.DOUBLE, BondType.TRIPLE]:
                other_idx = bond.GetOtherAtomIdx(atom.GetIdx())
                # Check if other atom has another multiple bond (conjugated system)
                for bond2 in mol.GetAtomWithIdx(other_idx).GetBonds():
                    if bond2.GetBondType() in [BondType.DOUBLE, BondType.TRIPLE] and bond2 != bond:
                        third_idx = bond2.GetOtherAtomIdx(other_idx)
                        third_lone_pairs = _get_lone_pairs(mol.GetAtomWithIdx(third_idx))
                        
                        # Create new structure with shifted lone pair and bond orders
                        new_mol = Chem.RWMol(mol)
                        # Just adjust bond orders and charges for now, skip lone pair manipulation
                        new_mol.GetBondBetweenAtoms(atom.GetIdx(), other_idx).SetBondType(BondType.DOUBLE)
                        new_mol.GetBondBetweenAtoms(other_idx, third_idx).SetBondType(BondType.SINGLE)
                        
                        # Update formal charges
                        new_mol.GetAtomWithIdx(atom.GetIdx()).SetFormalCharge(
                            new_mol.GetAtomWithIdx(atom.GetIdx()).GetFormalCharge() + 1)
                        new_mol.GetAtomWithIdx(third_idx).SetFormalCharge(
                            new_mol.GetAtomWithIdx(third_idx).GetFormalCharge() - 1)
                        
                        if Chem.Kekulize(new_mol):
                            structures.append(Molecule._from_rdmol(Chem.Mol(new_mol)))
    return structures


def _generate_adj_lone_pair_radical_resonance_structures(mol):
    """Generate resonance structures by adjacent lone pair-radical shift."""
    structures = []
    if not _is_radical(mol) or not _has_lone_pairs(mol):
        return structures
        
    # Find adjacent atoms where one has radical, other has lone pair
    for bond in mol.GetBonds():
        a1_idx = bond.GetBeginAtomIdx()
        a2_idx = bond.GetEndAtomIdx()
        
        a1 = mol.GetAtomWithIdx(a1_idx)
        a2 = mol.GetAtomWithIdx(a2_idx)
        
        # Check if a1 has radical and a2 has lone pair, or vice versa
        if (a1.GetNumRadicalElectrons() > 0 and _get_lone_pairs(a2) > 0) or \
           (a2.GetNumRadicalElectrons() > 0 and _get_lone_pairs(a1) > 0):
            
            # Determine direction
            if a1.GetNumRadicalElectrons() > 0 and _get_lone_pairs(a2) > 0:
                rad_idx, lp_idx = a1_idx, a2_idx
            else:
                rad_idx, lp_idx = a2_idx, a1_idx
            
            # Create new structure
            new_mol = Chem.RWMol(mol)
            
            # Move radical, adjust charges (skip lone pair manipulation)
            new_mol.GetAtomWithIdx(rad_idx).SetNumRadicalElectrons(0)
            new_mol.GetAtomWithIdx(lp_idx).SetNumRadicalElectrons(
                new_mol.GetAtomWithIdx(lp_idx).GetNumRadicalElectrons() + 1)
            
            # Update formal charges
            new_mol.GetAtomWithIdx(rad_idx).SetFormalCharge(
                new_mol.GetAtomWithIdx(rad_idx).GetFormalCharge() + 1)
            new_mol.GetAtomWithIdx(lp_idx).SetFormalCharge(
                new_mol.GetAtomWithIdx(lp_idx).GetFormalCharge() - 1)
            
            if Chem.Kekulize(new_mol):
                structures.append(Molecule._from_rdmol(Chem.Mol(new_mol)))
    return structures


def _generate_optimal_aromatic_resonance_structures(mol):
    """Generate optimal aromatic resonance structures for aromatic molecules."""
    if not _is_aromatic(mol):
        return []
        
    # Convert to aromatic form
    new_mol = Chem.RWMol(mol)
    try:
        Chem.Kekulize(new_mol, clearAromaticFlags=False)
        Chem.SetAromaticity(new_mol)
        return [Molecule._from_rdmol(Chem.Mol(new_mol))]
    except Exception:
        return []


def _generate_kekule_structure(mol):
    """Generate kekulized structure for aromatic molecules."""
    if not _is_aromatic(mol):
        return []
        
    try:
        new_mol = Chem.RWMol(mol)
        Chem.Kekulize(new_mol, clearAromaticFlags=True)
        return [Molecule._from_rdmol(Chem.Mol(new_mol))]
    except Exception:
        return []


def generate_resonance_structures(molecule: Molecule) -> list[Molecule]:
    """
    Generate resonance structures for a molecule using RMG-style rules.
    
    Args:
        molecule: A Molecule instance for which to generate resonance structures.
        
    Returns:
        A list of Molecule instances representing the resonance structures.
        The input molecule is included in the list.
    """
    results = [molecule.copy()]
    
    # Get RDKit molecule and analyze features
    rdmol = molecule._rdkit
    features = _analyze_molecule(rdmol)
    
    # Generate resonance structures using different algorithms
    new_structures = []
    
    if features['is_radical'] and not features['is_aromatic']:
        new_structures.extend(_generate_allyl_delocalization_resonance_structures(rdmol))
    
    if features['hasLonePairs'] and not features['is_aromatic']:
        new_structures.extend(_generate_lone_pair_multiple_bond_resonance_structures(rdmol))
        new_structures.extend(_generate_adj_lone_pair_radical_resonance_structures(rdmol))
    
    if features['is_aromatic']:
        new_structures.extend(_generate_optimal_aromatic_resonance_structures(rdmol))
        new_structures.extend(_generate_kekule_structure(rdmol))
    
    # Add new structures if they're different from the original
    for structure in new_structures:
        if structure.to_smiles() != molecule.to_smiles():
            results.append(structure)
    
    return results


# Expose this function as a method on Molecule
Molecule.get_resonance_structures = generate_resonance_structures
