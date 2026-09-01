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


def _get_lone_pairs(mol, atom):
    """Lone pairs per RMG-Py Molecule.update_lone_pairs():

        lone_pairs = (valence_electrons - radical_electrons - charge - bond_order) / 2

    Hydrogen (and lithium) always have 0. Matches rmgpu.molecule.adjlist.
    get_lone_pairs (the RMG-Py port).

    RMG-Py counts bond order on the EXPLICIT-H graph (Molecule.
    get_total_bond_order over its stored explicit-hydrogen vertices). rmgpu
    mols are stored implicit-H, so the bond order is computed after adding
    hydrogens. This is what keeps the lone-pair-radical generator from firing
    on pure hydrocarbon radicals: a [CH] radical carbon has total bond order
    4 once its implicit H is added, so LP = (4 - 1 - 0 - 4) < 0 -> 0, and no
    ionic resonance forms are emitted (RMG-Py never produces them). A per-
    element heuristic (the original) gave C/H nonzero lone pairs at all, which
    wrongly enabled the generator (a product-set divergence at the job-05 gate).
    """
    from rdkit import Chem
    from rmgpu.molecule.adjlist import VALENCE_ELECTRONS, _NO_LONE_PAIRS
    symbol = atom.GetSymbol()
    if symbol in _NO_LONE_PAIRS:
        return 0
    ve = VALENCE_ELECTRONS.get(symbol)
    if ve is None:
        return 0
    charge = atom.GetFormalCharge()
    unpaired = atom.GetNumRadicalElectrons()
    explicit = any(a.GetSymbol() == 'H' for a in mol.GetAtoms())
    if not explicit:
        try:
            em = Chem.AddHs(Chem.Mol(mol))
            eatom = em.GetAtomWithIdx(atom.GetIdx())
            bond_order = sum(b.GetBondTypeAsDouble() for b in eatom.GetBonds())
        except Exception:
            bond_order = sum(b.GetBondTypeAsDouble()
                             for b in atom.GetBonds())
    else:
        bond_order = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
    return max(0, (ve - unpaired - charge - int(bond_order)) // 2)


def _has_nitrogen_val5(mol):
    """Check for nitrogen valence 5 atoms without lone pairs."""
    return any(atom.GetSymbol() == 'N' and atom.GetFormalCharge() >= 1 for atom in mol.GetAtoms())


def _has_lone_pairs(mol):
    """Check if molecule has any atoms with lone pairs."""
    return any(_get_lone_pairs(mol, atom) > 0 for atom in mol.GetAtoms())


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
    """Generate resonance structures by allyl radical shift.

    A radical adjacent to a pi bond delocalizes: the radical moves to the far
    end of the pi bond and the pi bond becomes a single bond (the new pi bond
    forms between the original radical site and the near end). The pi bond is
    matched as an explicit DOUBLE bond OR an AROMATIC ring bond, so a radical
    exocyclic to an aromatic ring (a benzylic radical) also delocalizes into
    the ring, forming an exocyclic C=C and a radical on the ring - exactly
    the resonance forms RMG-Py generates for e.g. 1-phenylethyl.

    The radical atom itself must be NON-aromatic: shifting within an aromatic
    ring (an aryl radical) would break the ring's pi system, and RMG-Py does
    not generate those (the ring is already the delocalized form).
    """
    structures = []
    if not _is_radical(mol):
        return structures

    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() == 0:
            continue
        # An aryl radical (radical on an aromatic ring atom) is already the
        # delocalized form - do not shift within the ring.
        if atom.GetIsAromatic():
            continue

        rad_idx = atom.GetIdx()

        # The pi bond the radical delocalizes through must have the radical
        # bonded directly to one of its two atoms. In a kekulized aromatic
        # ring the ipso carbon (the ring atom the radical is bonded to) carries
        # ONE double and ONE single ring bond, but both are "pi bonds" in the
        # aromatic resonance - the radical delocalizes to BOTH ortho carbons.
        # So include every pi bond (explicit double or aromatic) adjacent to
        # the radical, PLUS the single ring bond of an aromatic ipso carbon.
        candidates = {}
        for b2 in atom.GetBonds():
            other_idx = b2.GetOtherAtomIdx(rad_idx)
            neighbor_atom = mol.GetAtomWithIdx(other_idx)
            for bond in neighbor_atom.GetBonds():
                if bond.GetBeginAtomIdx() == rad_idx or \
                        bond.GetEndAtomIdx() == rad_idx:
                    continue  # the C(radical)-C(bonded) bond itself
                bt = bond.GetBondType()
                is_pi = bt in (BondType.DOUBLE, BondType.AROMATIC)
                # A single ring bond of an aromatic ipso carbon is a valid
                # delocalization path (the other ortho resonance form).
                is_aromatic_ring_bond = (
                    not is_pi and bond.GetIsAromatic() and
                    neighbor_atom.GetIsAromatic())
                if is_pi or is_aromatic_ring_bond:
                    a1 = bond.GetBeginAtomIdx()
                    a2 = bond.GetEndAtomIdx()
                    # The bond must connect the radical to the far atom:
                    # one endpoint is the ipso (bonded) carbon, the other is
                    # the far atom the radical migrates to.
                    if other_idx in (a1, a2):
                        key = frozenset((a1, a2))
                        if key not in candidates:
                            candidates[key] = bond

        for bond in candidates.values():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()

            # Determine target (the atom opposite the radical on the pi bond).
            is_adjacent_a1 = atom.GetIdx() in (a1, a2) or a1 == rad_idx
            # The radical is bonded to one endpoint (the ipso carbon); the
            # target is the other.
            if a1 == rad_idx or a2 == rad_idx:
                continue  # can't happen (we excluded the radical's own bond)
            # The radical is NOT an endpoint of this bond (it's bonded to the
            # ipso carbon which IS an endpoint); identify which endpoint is
            # bonded to the radical.
            rad_bonded_endpoint = None
            for b2 in atom.GetBonds():
                o = b2.GetOtherAtomIdx(rad_idx)
                if o == a1 or o == a2:
                    rad_bonded_endpoint = o
                    break
            if rad_bonded_endpoint is None:
                continue
            if rad_bonded_endpoint == a1:
                target_idx = a2
                near_idx = a1
            else:
                target_idx = a1
                near_idx = a2

            # Create new structure with shifted radical and bonds.
            new_mol = Chem.RWMol(mol)

            # Move radical to target.
            new_mol.GetAtomWithIdx(target_idx).SetNumRadicalElectrons(
                new_mol.GetAtomWithIdx(target_idx).GetNumRadicalElectrons() + 1)
            new_mol.GetAtomWithIdx(rad_idx).SetNumRadicalElectrons(0)

            # The pi bond becomes single; a new pi bond forms between the
            # original radical site and the near end (forming the exocyclic
            # C=C when the pi bond was an aromatic ring bond).
            new_mol.GetBondBetweenAtoms(a1, a2).SetBondType(BondType.SINGLE)
            new_mol.GetBondBetweenAtoms(rad_idx, near_idx).SetBondType(
                BondType.DOUBLE)

            # Clear aromaticity on the touched bonds so kekulization sees the
            # explicit bond orders (the ring is now a kekulized ring bearing
            # an exocyclic double bond).
            for b in new_mol.GetBonds():
                b.SetIsAromatic(False)

            # Validate the structure. The radical landing on a ring carbon
            # leaves a stale valence cache, so sanitize first (otherwise
            # Kekulize reports "Unkekulized atoms" even for a valid shift);
            # an invalid shift - e.g. the radical landing on a fully-
            # substituted ring atom - still raises KekulizeException and the
            # candidate is dropped. This rdkit build's Kekulize returns None
            # (not True) on success, so a raised exception is the only failure
            # signal.
            try:
                Chem.SanitizeMol(new_mol)
                Chem.Kekulize(new_mol, clearAromaticFlags=True)
            except Exception:
                pass
            else:
                structures.append(Molecule._from_rdmol(Chem.Mol(new_mol)))
    return structures


def _generate_lone_pair_multiple_bond_resonance_structures(mol):
    """Generate resonance structures by lone pair shift with double/triple bond."""
    structures = []
    if not _has_lone_pairs(mol):
        return structures
        
    # Find 3-atom systems where lone pair can shift to multiple bond
    for atom in mol.GetAtoms():
        if _get_lone_pairs(mol, atom) == 0:
            continue
            
        # Look for adjacent atoms with multiple bonds
        for bond in atom.GetBonds():
            if bond.GetBondType() in [BondType.DOUBLE, BondType.TRIPLE]:
                other_idx = bond.GetOtherAtomIdx(atom.GetIdx())
                # Check if other atom has another multiple bond (conjugated system)
                for bond2 in mol.GetAtomWithIdx(other_idx).GetBonds():
                    if bond2.GetBondType() in [BondType.DOUBLE, BondType.TRIPLE] and bond2 != bond:
                        third_idx = bond2.GetOtherAtomIdx(other_idx)
                        third_lone_pairs = _get_lone_pairs(mol, mol.GetAtomWithIdx(third_idx))
                        
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
        if (a1.GetNumRadicalElectrons() > 0 and _get_lone_pairs(mol, a2) > 0) or \
           (a2.GetNumRadicalElectrons() > 0 and _get_lone_pairs(mol, a1) > 0):
            
            # Determine direction
            if a1.GetNumRadicalElectrons() > 0 and _get_lone_pairs(mol, a2) > 0:
                rad_idx, lp_idx = a1_idx, a2_idx
            else:
                rad_idx, lp_idx = a2_idx, a1_idx
            
            # Create new structure
            new_mol = Chem.RWMol(mol)

            # Move the radical from the radical site to the lone-pair site.
            # The radical site gains a lone pair (charge decreases by 1) and
            # the lone-pair site gains the radical (charge increases by 1).
            # This mirrors RMG-Py's adj lone-pair-radical resonance transition.
            new_mol.GetAtomWithIdx(rad_idx).SetNumRadicalElectrons(0)
            new_mol.GetAtomWithIdx(lp_idx).SetNumRadicalElectrons(
                new_mol.GetAtomWithIdx(lp_idx).GetNumRadicalElectrons() + 1)

            # Update formal charges (rad site -1, lone-pair site +1)
            new_mol.GetAtomWithIdx(rad_idx).SetFormalCharge(
                new_mol.GetAtomWithIdx(rad_idx).GetFormalCharge() - 1)
            new_mol.GetAtomWithIdx(lp_idx).SetFormalCharge(
                new_mol.GetAtomWithIdx(lp_idx).GetFormalCharge() + 1)

            try:
                new_mol.UpdatePropertyCache()
                Chem.SanitizeMol(new_mol)
            except Exception:
                continue
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

    # Allyl radical delocalization. The radical atom is guarded inside
    # _generate_allyl_delocalization_resonance_structures to be non-aromatic,
    # so in-ring (aryl) radicals do not break their ring's pi system, while
    # exocyclic radicals (a benzylic radical) delocalize into an adjacent
    # aromatic ring (forming the exocyclic C=C + ring-radical forms RMG-Py
    # generates). The old `and not is_aromatic` gate suppressed this for
    # aromatic molecules and dropped those exocyclic forms (an Intra_ene
    # product-set divergence at the job-05 gate), so it is now always run for
    # radicals.
    if features['is_radical']:
        new_structures.extend(
            _generate_allyl_delocalization_resonance_structures(rdmol))

    if features['hasLonePairs'] and not features['is_aromatic']:
        new_structures.extend(_generate_lone_pair_multiple_bond_resonance_structures(rdmol))
        new_structures.extend(_generate_adj_lone_pair_radical_resonance_structures(rdmol))

    if features['is_aromatic']:
        new_structures.extend(_generate_optimal_aromatic_resonance_structures(rdmol))
        new_structures.extend(_generate_kekule_structure(rdmol))

    # Deduplicate against the input (canonical SMILES)
    all_structures = [molecule.copy()]
    input_smi = molecule.to_smiles()
    for structure in new_structures:
        smi = structure.to_smiles()
        if smi != input_smi:
            all_structures.append(structure)

    # Filtration: keep only representative structures (RMG-Py
    # filtration.filter_structures), always preserving the input.
    from rmgpu.molecule.resonance_filtration import filter_structures
    return filter_structures(all_structures, molecule.copy())


# Expose this function as a method on Molecule
Molecule.get_resonance_structures = generate_resonance_structures
