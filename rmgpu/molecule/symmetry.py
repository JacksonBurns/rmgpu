"""
Symmetry number estimation for rmgpu, ported from RMG-Py symmetry.py.

The algorithm multiplies contributions from:
  - each non-cyclic atom (calculate_atom_symmetry_number)
  - each non-cyclic bond, once per bond (calculate_bond_symmetry_number)
  - cumulated double-bond axes (calculate_axis_symmetry_number)
  - cyclic regions (calculate_cyclic_symmetry_number)

All factors are computed on the graph with explicit hydrogen atoms, which is
how RMG-Py represents a molecule.
"""

import itertools
from collections import deque

from rdkit import Chem
from rdkit.Chem import rdchem


class _SymMol:
    """Duck-typed carrier so the ported functions can use mol._rdkit."""

    def __init__(self, rdmol):
        self._rdkit = rdmol


def _sym_graph(molecule):
    """
    Return an object exposing the graph RMG-Py would use for symmetry
    calculation: explicit hydrogens, kekulized.
    """
    if hasattr(molecule, '_with_explicit_h'):
        rdmol = molecule._with_explicit_h()
    else:  # already a bare RDKit mol (testing)
        rdmol = molecule
    return _SymMol(rdmol)


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
            for i in range(len(ring_list)):
                a, b = ring_list[i], ring_list[(i + 1) % len(ring_list)]
                if (a == idx1 and b == idx2) or (a == idx2 and b == idx1):
                    return True
    return False


def _split_into_groups(rdmol):
    """
    Split an RDKit mol into connected components.

    Returns a list of (Molecule, index_map) where index_map maps old atom
    indices to new atom indices inside the fragment.
    """
    from rmgpu.molecule.molecule import Molecule

    if rdmol is None or rdmol.GetNumAtoms() == 0:
        return []

    visited = set()
    groups = []

    for atom_idx in range(rdmol.GetNumAtoms()):
        if atom_idx in visited:
            continue
        queue = deque([atom_idx])
        component = []
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for bond in rdmol.GetAtomWithIdx(current).GetBonds():
                other_idx = bond.GetOtherAtom(rdmol.GetAtomWithIdx(current)).GetIdx()
                if other_idx not in visited:
                    queue.append(other_idx)

        rw_new = Chem.RWMol()
        idx_map = {}
        for atom_idx in component:
            atom = rdmol.GetAtomWithIdx(atom_idx)
            new_atom = Chem.Atom(atom.GetSymbol())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons())
            idx_map[atom_idx] = rw_new.AddAtom(new_atom)

        added_bonds = set()
        for atom_idx in component:
            for bond in rdmol.GetAtomWithIdx(atom_idx).GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                if begin_idx not in idx_map or end_idx not in idx_map:
                    continue
                bond_key = (min(begin_idx, end_idx), max(begin_idx, end_idx))
                if bond_key in added_bonds:
                    continue
                added_bonds.add(bond_key)
                rw_new.AddBond(idx_map[begin_idx], idx_map[end_idx], bond.GetBondType())

        mol = Molecule._from_rdmol(Chem.Mol(rw_new))
        groups.append((mol, idx_map))

    return groups


def _check_isomorphism(mol1, mol2):
    """Check if two molecules are isomorphic using canonical SMILES."""
    if mol1 is None or mol2 is None:
        return False
    try:
        return Chem.MolToSmiles(mol1._rdkit) == Chem.MolToSmiles(mol2._rdkit)
    except Exception:
        return False


def calculate_atom_symmetry_number(molecule, atom_idx):
    """
    Return the symmetry number centered at `atom_idx` in the structure.
    The atom of interest must not be in a cycle. (Ported from RMG-Py.)
    """
    symmetry_number = 1
    rdmol = molecule._rdkit
    atom = rdmol.GetAtomWithIdx(atom_idx)

    single = double = triple = benzene = num_neighbors = 0
    for bond in atom.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type == rdchem.BondType.SINGLE:
            single += 1
        elif bond_type == rdchem.BondType.DOUBLE:
            double += 1
        elif bond_type == rdchem.BondType.TRIPLE:
            triple += 1
        elif bond_type == rdchem.BondType.AROMATIC:
            benzene += 1
        num_neighbors += 1

    # If atom has zero or one neighbors, the symmetry number is 1
    if num_neighbors < 2:
        return symmetry_number

    # Remove the atom and split into groups
    rwmol = Chem.RWMol(rdmol)
    rwmol.RemoveAtom(atom_idx)
    remaining = Chem.Mol(rwmol)
    groups = [g for g, _ in _split_into_groups(remaining)]

    # Determine equivalence of functional groups around atom
    group_isomorphism = {i: {} for i in range(len(groups))}
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i != j and j not in group_isomorphism[i]:
                group_isomorphism[i][j] = _check_isomorphism(group1, group2)
                group_isomorphism[j][i] = group_isomorphism[i][j]
            elif i == j:
                group_isomorphism[i][i] = True

    count = [sum([int(group_isomorphism[i][j]) for j in range(len(groups))])
             for i in range(len(groups))]
    for _ in range(count.count(2) // 2):
        count.remove(2)
    for _ in range(count.count(3) // 3):
        count.remove(3)
        count.remove(3)
    for _ in range(count.count(4) // 4):
        count.remove(4)
        count.remove(4)
        count.remove(4)
    count.sort()
    count.reverse()

    radical_electrons = atom.GetNumRadicalElectrons()

    if radical_electrons == 0:
        if single == 4:
            # Four single bonds
            if count == [4]:
                symmetry_number *= 12
            elif count == [3, 1]:
                symmetry_number *= 3
            elif count == [2, 2]:
                symmetry_number *= 2
            elif count == [2, 1, 1]:
                symmetry_number *= 1
            elif count == [1, 1, 1, 1]:
                symmetry_number *= 0.5  # found chirality
        elif single == 3:
            # Three single bonds
            if count == [3]:
                symmetry_number *= 3
            elif count == [2, 1]:
                symmetry_number *= 1
            elif count == [1, 1, 1]:
                symmetry_number *= 1
        elif single == 2:
            # Two single bonds
            if count == [2]:
                symmetry_number *= 2
        # for resonance hybrids
        elif single == 1:
            if count == [2, 1]:
                symmetry_number *= 2
        elif double == 2:
            # Two double bonds
            if count == [2]:
                symmetry_number *= 2
        # for nitrogen resonance hybrids
        elif single == 0:
            if count == [2]:
                symmetry_number *= 2
    elif radical_electrons == 1:
        if single == 3:
            # Three single bonds
            if count == [3]:
                symmetry_number *= 6
            elif count == [2, 1]:
                symmetry_number *= 2
            elif count == [1, 1, 1]:
                symmetry_number *= 1
        elif single == 1:
            if count == [2, 1]:
                symmetry_number *= 2
            elif count == [1, 1, 1]:
                symmetry_number *= 1
    elif radical_electrons == 2:
        if single == 2:
            # Two single bonds
            if count == [2]:
                symmetry_number *= 2

    return symmetry_number


def _remove_bond_and_split(rdmol, atom1_idx, atom2_idx):
    """
    Remove the bond (atom1_idx, atom2_idx) from a copy of rdmol and return
    (structure_rdmol, atom1_in_structure, atom2_in_structure) with atom
    positions in the structure. The removed bond's two ends keep their
    relative atom indices (RemoveBond does not renumber).
    """
    rwmol = Chem.RWMol(rdmol)
    rwmol.RemoveBond(atom1_idx, atom2_idx)
    return Chem.Mol(rwmol)


def calculate_bond_symmetry_number(molecule, atom1_idx, atom2_idx):
    """
    Return the symmetry number centered at the bond between atom1_idx and
    atom2_idx. (Ported from RMG-Py; atoms compared by type/charge/radicals.)
    """
    rdmol = molecule._rdkit
    atom1 = rdmol.GetAtomWithIdx(atom1_idx)
    atom2 = rdmol.GetAtomWithIdx(atom2_idx)
    symmetry_number = 1

    # RMG-Py requires the atoms to be of equivalent type
    if atom1.GetSymbol() != atom2.GetSymbol():
        return symmetry_number
    if atom1.GetFormalCharge() != atom2.GetFormalCharge():
        return symmetry_number
    if atom1.GetNumRadicalElectrons() != atom2.GetNumRadicalElectrons():
        return symmetry_number

    # An O-O bond is considered to be an "optical isomer" and so no symmetry
    # correction will be applied
    if (atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'O'
            and atom1.GetNumRadicalElectrons() == 0
            and atom2.GetNumRadicalElectrons() == 0):
        return symmetry_number

    # If the molecule is diatomic, then we don't have to check the ligands
    # on the two atoms in this bond (since we know there aren't any)
    if rdmol.GetNumAtoms() == 2:
        return 2

    # Remove the bond and split into fragments
    structure = _remove_bond_and_split(rdmol, atom1_idx, atom2_idx)
    fragments = _split_into_groups(structure)

    if len(fragments) != 2:
        return symmetry_number

    # Determine which fragment contains each atom
    atom1_frag = None
    atom2_frag = None
    for mol, idx_map in fragments:
        if atom1_idx in idx_map:
            atom1_frag = (mol, idx_map)
        if atom2_idx in idx_map:
            atom2_frag = (mol, idx_map)

    if atom1_frag is None or atom2_frag is None:
        return symmetry_number

    fragment1, map1 = atom1_frag
    fragment2, map2 = atom2_frag

    # Build each fragment with the bonded atom removed, then split into groups
    def _fragment_minus_atom(idx_map, old_idx):
        rwmol = Chem.RWMol()
        new_map = {}
        old_indices = [i for i in idx_map if i != old_idx]
        for old_i in old_indices:
            src = molecule._rdkit.GetAtomWithIdx(old_i)
            a = Chem.Atom(src.GetSymbol())
            a.SetFormalCharge(src.GetFormalCharge())
            a.SetNumRadicalElectrons(src.GetNumRadicalElectrons())
            new_map[old_i] = rwmol.AddAtom(a)
        added = set()
        for old_i in old_indices:
            for bond in molecule._rdkit.GetAtomWithIdx(old_i).GetBonds():
                b1 = bond.GetBeginAtomIdx()
                b2 = bond.GetEndAtomIdx()
                if b1 not in new_map or b2 not in new_map:
                    continue
                key = (min(b1, b2), max(b1, b2))
                if key in added:
                    continue
                added.add(key)
                rwmol.AddBond(new_map[b1], new_map[b2], bond.GetBondType())
        return Chem.Mol(rwmol)

    frag1_minus = _fragment_minus_atom(map1, atom1_idx)
    frag2_minus = _fragment_minus_atom(map2, atom2_idx)

    groups1 = [g for g, _ in _split_into_groups(frag1_minus)]
    groups2 = [g for g, _ in _split_into_groups(frag2_minus)]

    # Test functional groups for symmetry (ported from RMG-Py)
    if len(groups1) == len(groups2) == 1:
        if _check_isomorphism(groups1[0], groups2[0]):
            symmetry_number *= 2
    elif len(groups1) == len(groups2) == 2:
        if _check_isomorphism(groups1[0], groups2[0]) and _check_isomorphism(groups1[1], groups2[1]):
            symmetry_number *= 2
        elif _check_isomorphism(groups1[1], groups2[0]) and _check_isomorphism(groups1[0], groups2[1]):
            symmetry_number *= 2
    elif len(groups1) == len(groups2) == 3:
        if _check_isomorphism(groups1[0], groups2[0]):
            if _check_isomorphism(groups1[1], groups2[1]) and _check_isomorphism(groups1[2], groups2[2]):
                symmetry_number *= 2
            elif _check_isomorphism(groups1[1], groups2[2]) and _check_isomorphism(groups1[2], groups2[1]):
                symmetry_number *= 2
        elif _check_isomorphism(groups1[0], groups2[1]):
            if _check_isomorphism(groups1[1], groups2[2]) and _check_isomorphism(groups1[2], groups2[0]):
                symmetry_number *= 2
            elif _check_isomorphism(groups1[1], groups2[0]) and _check_isomorphism(groups1[2], groups2[2]):
                symmetry_number *= 2
        elif _check_isomorphism(groups1[0], groups2[2]):
            if _check_isomorphism(groups1[1], groups2[1]) and _check_isomorphism(groups1[2], groups2[0]):
                symmetry_number *= 2
            elif _check_isomorphism(groups1[1], groups2[0]) and _check_isomorphism(groups1[2], groups2[1]):
                symmetry_number *= 2

    return symmetry_number


def _subgraph(rdmol, atom_set, remove_bonds=()):
    """
    Build a new RDKit mol containing only `atom_set` (original indices), with
    `remove_bonds` (original index pairs) excluded. Returns (new_rdmol, map)
    where map maps original index -> new index.
    """
    rwmol = Chem.RWMol()
    idx_map = {}
    for old_i in atom_set:
        src = rdmol.GetAtomWithIdx(old_i)
        a = Chem.Atom(src.GetSymbol())
        a.SetFormalCharge(src.GetFormalCharge())
        a.SetNumRadicalElectrons(src.GetNumRadicalElectrons())
        idx_map[old_i] = rwmol.AddAtom(a)
    removed = set()
    for b1, b2 in remove_bonds:
        removed.add((min(b1, b2), max(b1, b2)))
    added = set()
    for old_i in atom_set:
        for bond in rdmol.GetAtomWithIdx(old_i).GetBonds():
            b1 = bond.GetBeginAtomIdx()
            b2 = bond.GetEndAtomIdx()
            if b1 not in idx_map or b2 not in idx_map:
                continue
            key = (min(b1, b2), max(b1, b2))
            if key in removed or key in added:
                continue
            added.add(key)
            rwmol.AddBond(idx_map[b1], idx_map[b2], bond.GetBondType())
    return Chem.Mol(rwmol), idx_map


def calculate_axis_symmetry_number(molecule):
    """
    Get the axis symmetry number correction. The "axis" refers to a series
    of two or more cumulated double bonds (e.g. C=C=C, etc.). (Ported.)
    """
    from rmgpu.molecule.atomtype import match_atom_type, _get_atom_features

    symmetry_number = 1
    rdmol = molecule._rdkit
    ri = rdmol.GetRingInfo()
    all_atoms = set(range(rdmol.GetNumAtoms()))

    # List all double bonds in the structure
    double_bonds = []
    for atom1_idx in range(rdmol.GetNumAtoms()):
        atom1 = rdmol.GetAtomWithIdx(atom1_idx)
        for bond in atom1.GetBonds():
            atom2_idx = bond.GetOtherAtom(atom1).GetIdx()
            if atom2_idx <= atom1_idx:
                continue
            if bond.GetBondType() == rdchem.BondType.DOUBLE:
                double_bonds.append((atom1_idx, atom2_idx))

    # Search for adjacent double bonds
    cumulated_bonds = []
    for i, bond1 in enumerate(double_bonds):
        atom11, atom12 = bond1
        for bond2 in double_bonds[i + 1:]:
            atom21, atom22 = bond2
            if atom11 == atom21 or atom11 == atom22 or atom12 == atom21 or atom12 == atom22:
                list_to_add_to = None
                for cum_bonds in cumulated_bonds:
                    if (atom11, atom12) in cum_bonds or (atom21, atom22) in cum_bonds:
                        list_to_add_to = cum_bonds
                if list_to_add_to is not None:
                    if (atom11, atom12) not in list_to_add_to:
                        list_to_add_to.append((atom11, atom12))
                    if (atom21, atom22) not in list_to_add_to:
                        list_to_add_to.append((atom21, atom22))
                else:
                    cumulated_bonds.append([(atom11, atom12), (atom21, atom22)])

    # Also keep isolated double bonds
    for bond1 in double_bonds:
        for bonds in cumulated_bonds:
            if bond1 in bonds:
                break
        else:
            cumulated_bonds.append([bond1])

    # For each set of adjacent double bonds, check for axis symmetry
    for bonds in cumulated_bonds:
        # Do nothing if axis is in cycle
        found = False
        for atom1_idx, atom2_idx in bonds:
            if _is_bond_in_ring(ri, atom1_idx, atom2_idx):
                found = True
                break
        if found:
            continue

        # Find terminal atoms in axis
        axis = []
        for bond in bonds:
            axis.extend(bond)
        terminal_atoms = [a for a in set(axis) if axis.count(a) == 1]
        if len(terminal_atoms) != 2:
            continue

        # Remove axis bonds, drop atoms no longer bonded to anything
        structure, s_map = _subgraph(rdmol, all_atoms, remove_bonds=bonds)
        orig_of_s1 = {new: old for old, new in s_map.items()}  # s1 index -> original
        isolated = [new_i for new_i in range(structure.GetNumAtoms())
                    if len(structure.GetAtomWithIdx(new_i).GetBonds()) == 0]
        # keep only isolated atoms that are NOT axis terminals
        keep = set(range(structure.GetNumAtoms()))
        for new_i in isolated:
            if orig_of_s1[new_i] not in terminal_atoms:
                keep.discard(new_i)
        if len(keep) != structure.GetNumAtoms():
            structure, s_map2 = _subgraph(structure, keep)
            orig_of_struct = {s_map2[new]: orig for new, orig in
                             ((k, orig_of_s1[k]) for k in s_map2)}
        else:
            orig_of_struct = dict(orig_of_s1)  # structure index -> original

        # Split remaining fragments of structure
        end_fragments = _split_into_groups(structure)

        # there can be two groups at each end of the axis
        symmetry_broken = False
        end_fragments_to_remove = []
        for mol, idx_map in end_fragments:
            # find the terminal atom in this fragment
            terminal_struct_idx = None
            for struct_i in idx_map:
                if orig_of_struct[struct_i] in terminal_atoms:
                    terminal_struct_idx = struct_i
                    break
            if terminal_struct_idx is None:
                continue

            # split what's left (after removing the terminal atom) into groups
            frag_minus, f_map = _subgraph(structure, set(idx_map) - {terminal_struct_idx})
            groups = [g for g, _ in _split_into_groups(frag_minus)]

            terminal_atom = rdmol.GetAtomWithIdx(orig_of_struct[terminal_struct_idx])

            # If end has only one group it can't contribute to (nor break)
            # axial symmetry, unless it is an N3d-type nitrogen
            if len(groups) == 0:
                end_fragments_to_remove.append((mol, idx_map))
                continue
            elif len(groups) == 1 and terminal_atom.GetNumRadicalElectrons() == 0:
                feat = _get_atom_features(terminal_atom)
                if match_atom_type(terminal_atom.GetSymbol(), feat) == 'N3d':
                    symmetry_broken = True
                else:
                    end_fragments_to_remove.append((mol, idx_map))
                    continue
            elif len(groups) == 1 and terminal_atom.GetNumRadicalElectrons() != 0:
                symmetry_broken = True
            elif len(groups) == 2:
                if not _check_isomorphism(groups[0], groups[1]):
                    # this end has broken the symmetry of the axis
                    symmetry_broken = True

        for frag in end_fragments_to_remove:
            end_fragments.remove(frag)

        # If there are end fragments left that can contribute to symmetry,
        # and none of them broke it, then double the symmetry number
        if end_fragments and not symmetry_broken:
            symmetry_number *= 2

    return symmetry_number


def _fragment_minus_from(structure, idx_map, old_idx):
    """Build the fragment (atoms of idx_map minus old_idx) as a new RDKit mol."""
    return _subgraph(structure, set(idx_map) - {old_idx})[0]


def _indistinguishable(atom1, atom2):
    """
    Determine if two atoms are feasibly indistinguishable based on connections
    to nearest neighbors. (Ported from RMG-Py _indistinguishable.)
    """
    if atom1.GetSymbol() != atom2.GetSymbol():
        return False
    if atom1.GetFormalCharge() != atom2.GetFormalCharge():
        return False
    if atom1.GetNumRadicalElectrons() != atom2.GetNumRadicalElectrons():
        return False

    bond_orders_1 = sorted([bond.GetBondTypeAsDouble() for bond in atom1.GetBonds()])
    bond_orders_2 = sorted([bond.GetBondTypeAsDouble() for bond in atom2.GetBonds()])
    if bond_orders_1 != bond_orders_2:
        return False

    bonds_1 = list(atom1.GetBonds())
    bonds_2 = list(atom2.GetBonds())

    for i in range(len(bonds_1)):
        neighbor1 = bonds_1[i].GetOtherAtom(atom1)
        matched = False
        for j in range(len(bonds_2)):
            neighbor2 = bonds_2[j].GetOtherAtom(atom2)
            bond1 = bonds_1[i]
            bond2 = bonds_2[j]
            if (bond1 is not None and bond2 is not None
                    and bond1.GetBondType() == bond2.GetBondType()
                    and neighbor1.GetSymbol() == neighbor2.GetSymbol()
                    and neighbor1.GetFormalCharge() == neighbor2.GetFormalCharge()
                    and neighbor1.GetNumRadicalElectrons() == neighbor2.GetNumRadicalElectrons()):
                matched = True
                bonds_2 = bonds_2[:j] + bonds_2[j + 1:]
                break
        if not matched:
            return False

    return True


def calculate_cyclic_symmetry_number(molecule):
    """
    Get the symmetry number correction for cyclic regions of a molecule.
    (Ported from RMG-Py calculate_cyclic_symmetry_number.)
    """
    symmetry_number = 1
    rdmol = molecule._rdkit
    ri = rdmol.GetRingInfo()

    single_rings = [list(ring) for ring in ri.AtomRings()]

    for ring in single_rings:
        size = len(ring)

        # look for twisting rotation
        for num_sections in range(size, 0, -1):
            # only go through if it can give symmetry (only factors of size)
            if size % num_sections == 0:
                num_rotations = size // num_sections
                all_the_same = True
                starting_index = 0
                while all_the_same and starting_index < size // 2:
                    for atom_index in range(num_rotations, size, num_rotations):
                        if not _indistinguishable(
                                rdmol.GetAtomWithIdx(ring[starting_index]),
                                rdmol.GetAtomWithIdx(ring[(starting_index + atom_index) % size])):
                            all_the_same = False
                            break
                    starting_index += 1
                if all_the_same:
                    symmetry_number *= num_sections
                    break

        # look for flipping rotation
        if size % 2 == 0:
            flipping_atom_indexes = list(range(size // 2))
        else:
            flipping_atom_indexes = list(range(size))

        for flipping_atom_index in flipping_atom_indexes:
            # check for flipping across an axis containing atoms
            all_the_same = True
            min_index = flipping_atom_index + 1
            max_index = flipping_atom_index + size - 1
            while min_index <= max_index:
                if not _indistinguishable(
                        rdmol.GetAtomWithIdx(ring[min_index % size]),
                        rdmol.GetAtomWithIdx(ring[max_index % size])):
                    all_the_same = False
                    break
                min_index += 1
                max_index -= 1

            # check to make sure that the groups are identical on centers of
            # flipping
            if all_the_same:
                ringed_atom_ids = set(ring)
                if size % 2 == 0:  # look at two atoms
                    atom1 = rdmol.GetAtomWithIdx(ring[flipping_atom_index])
                    atom2 = rdmol.GetAtomWithIdx(ring[flipping_atom_index + size // 2])
                    non_ring_bonded_atoms = [
                        atom.GetNeighbors()[k]
                        for atom in (atom1, atom2)
                        for k in range(len(atom.GetNeighbors()))
                        if atom.GetNeighbors()[k].GetIdx() not in ringed_atom_ids
                    ]
                    if len(non_ring_bonded_atoms) < 3:
                        pass
                    elif len(non_ring_bonded_atoms) == 3:
                        identical = _indistinguishable(non_ring_bonded_atoms[0], non_ring_bonded_atoms[1])
                        identical2 = _indistinguishable(non_ring_bonded_atoms[0], non_ring_bonded_atoms[2])
                        identical3 = _indistinguishable(non_ring_bonded_atoms[1], non_ring_bonded_atoms[2])
                        if not (identical or identical2 or identical3):
                            all_the_same = False
                    elif len(non_ring_bonded_atoms) == 4:
                        same_sides = _indistinguishable(non_ring_bonded_atoms[0], non_ring_bonded_atoms[1]) and \
                                     _indistinguishable(non_ring_bonded_atoms[2], non_ring_bonded_atoms[3])
                        if not same_sides:
                            atom0_matching = _indistinguishable(non_ring_bonded_atoms[0], non_ring_bonded_atoms[2]) or \
                                              _indistinguishable(non_ring_bonded_atoms[0], non_ring_bonded_atoms[3])
                            atom1_matching = _indistinguishable(non_ring_bonded_atoms[1], non_ring_bonded_atoms[2]) or \
                                              _indistinguishable(non_ring_bonded_atoms[1], non_ring_bonded_atoms[3])
                            if not (atom0_matching and atom1_matching):
                                all_the_same = False
                else:
                    atom = rdmol.GetAtomWithIdx(ring[flipping_atom_index])
                    neighbors = atom.GetNeighbors()
                    non_ring_bonded_atoms = [n for n in neighbors
                                             if n.GetIdx() not in ringed_atom_ids]
                    if len(non_ring_bonded_atoms) < 2:
                        pass
                    elif len(non_ring_bonded_atoms) == 2:
                        identical = _indistinguishable(non_ring_bonded_atoms[0], non_ring_bonded_atoms[1])
                        if not identical:
                            # flipping a tetrahedral will not work
                            all_the_same = False

            # for even rings, check for flipping across bonds too
            if not all_the_same and size % 2 == 0:
                all_the_same = True
                min_index = flipping_atom_index
                max_index = flipping_atom_index + size - 1
                while min_index < max_index:
                    if not _indistinguishable(
                            rdmol.GetAtomWithIdx(ring[min_index % size]),
                            rdmol.GetAtomWithIdx(ring[max_index % size])):
                        all_the_same = False
                        break
                    min_index += 1
                    max_index -= 1

            if all_the_same:
                symmetry_number *= 2
                break

    return symmetry_number


def get_symmetry_number(molecule):
    """
    Return the symmetry number for the structure (external + internal modes).

    Computed on the explicit-hydrogen graph, matching RMG-Py:

        for each non-cyclic atom:     * atom factor
        for each non-cyclic bond (once each): * bond factor
        * axis factor
        * cyclic factor (if cyclic)
    """
    graph = _sym_graph(molecule)
    rdmol = graph._rdkit
    ri = rdmol.GetRingInfo()

    symmetry_number = 1

    # Atom contributions
    for atom_idx in range(rdmol.GetNumAtoms()):
        if not _is_in_ring(ri, atom_idx):
            symmetry_number *= calculate_atom_symmetry_number(graph, atom_idx)

    # Bond contributions (once per bond, i < j)
    for atom1_idx in range(rdmol.GetNumAtoms()):
        atom1 = rdmol.GetAtomWithIdx(atom1_idx)
        for bond in atom1.GetBonds():
            atom2_idx = bond.GetOtherAtom(atom1).GetIdx()
            if atom2_idx > atom1_idx and not _is_bond_in_ring(ri, atom1_idx, atom2_idx):
                symmetry_number *= calculate_bond_symmetry_number(
                    graph, atom1_idx, atom2_idx)

    # Axis contribution
    symmetry_number *= calculate_axis_symmetry_number(graph)

    # Cyclic contribution
    if ri.NumRings() > 0:
        symmetry_number *= calculate_cyclic_symmetry_number(graph)

    return symmetry_number
