"""
Atom type database and assignment for rmgpu.

Reimplements RMG-Py's atom type assignment (rmgpy/molecule/atomtype.py) on
rdgpu's RDKit-based molecule. Atom types are matched on the graph with
explicit hydrogen atoms, which is how RMG-Py represents a molecule, and an
atom's feature vector (single / all_double / r_double / o_double / s_double /
triple / quadruple / benzene / lone_pairs / charge) is compared against the
specific atom types for its element. An empty feature list in a type
definition is a wildcard (matches anything).

The matching is first-match in RMG-Py's specific-type order, so the order of
entries in _SPECIFIC below must mirror RMG-Py's ATOMTYPES[...].specific lists.
"""

from rdkit import Chem
from rdkit.Chem import rdchem

# Feature key order must match RMG-Py get_features():
_FEATURE_KEYS = (
    'single', 'all_double', 'r_double', 'o_double', 's_double',
    'triple', 'quadruple', 'benzene', 'lone_pairs', 'charge',
)


def _features(label, single, all_double, r_double, o_double, s_double,
              triple, quadruple, benzene, lone_pairs, charge):
    return {
        'label': label,
        'single': single,
        'all_double': all_double,
        'r_double': r_double,
        'o_double': o_double,
        's_double': s_double,
        'triple': triple,
        'quadruple': quadruple,
        'benzene': benzene,
        'lone_pairs': lone_pairs,
        'charge': charge,
    }


# Specific atom types per element, in RMG-Py first-match order.
# Empty list = wildcard for that feature.
_SPECIFIC = {
    'H': [
        _features('H0', [0, 1], [0], [0], [0], [0], [0], [0], [0], [0], [0]),
        _features('H+', [0], [0], [0], [0], [0], [0], [0], [0], [0], [+1]),
    ],
    'C': [
        _features('Ca', [0], [0], [], [], [], [0], [], [0], [2], [0]),
        _features('Cs', [0, 1, 2, 3, 4], [0], [0], [0], [0], [0], [], [0], [0], [0]),
        _features('Csc', [0, 1, 2, 3], [0], [], [], [], [0], [], [0], [0], [+1]),
        _features('Cd', [0, 1, 2], [1], [1], [0], [0], [0], [], [0], [0], [0]),
        _features('CO', [0, 1, 2], [1], [0], [1], [0], [0], [], [0], [0], [0]),
        _features('CS', [0, 1, 2], [1], [0], [0], [1], [0], [], [0], [0], [0]),
        _features('Cdd', [0], [2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0], [], [0], [0], [0]),
        _features('Cdc', [0, 1], [1], [0, 1], [0, 1], [0, 1], [0], [], [0], [0], [+1]),
        _features('Ctc', [0], [0], [0], [0], [0], [1], [], [0], [0], [+1]),
        _features('Ct', [0, 1], [0], [], [], [], [1], [], [0], [0], [0]),
        _features('Cb', [0, 1], [0], [], [], [], [0], [], [1, 2], [], []),
        _features('Cbf', [0], [0], [], [], [], [0], [], [3], [], []),
        _features('Cq', [0], [0], [0], [0], [0], [0], [1], [0], [], []),
        _features('C2s', [0, 1, 2], [0], [], [], [], [0], [], [0], [1], [0]),
        _features('C2sc', [0, 1, 2, 3], [0], [], [], [], [0], [], [0], [1], [-1]),
        _features('C2d', [0], [1], [], [], [], [0], [], [0], [1], [0]),
        _features('C2dc', [0, 1], [1], [], [], [], [0], [], [0], [1], [-1]),
        _features('C2tc', [0], [0], [0], [0], [0], [1], [], [0], [1], [-1]),
    ],
    'N': [
        _features('N0sc', [0, 1], [0], [0], [0], [0], [0], [], [0], [3], [-2]),
        _features('N1s', [0, 1], [0], [0], [0], [0], [0], [], [0], [2], [0]),
        _features('N1sc', [0, 1, 2], [0], [], [], [], [0], [], [0], [2], [-1]),
        _features('N1dc', [0], [1], [], [], [], [0], [], [0], [2], [-1]),
        _features('N3s', [0, 1, 2, 3], [0], [0], [0], [0], [0], [], [0], [1], [0]),
        _features('N3sc', [0, 1, 2], [0], [0], [0], [0], [0], [], [0], [1], [+1]),
        _features('N3d', [0, 1], [1], [], [], [], [0], [], [0], [1], [0]),
        _features('N3t', [0], [0], [0], [0], [0], [1], [], [0], [1], [0]),
        _features('N3b', [0], [0], [0], [0], [0], [0], [], [2], [1], [0]),
        _features('N5sc', [0, 1, 2, 3, 4], [0], [0], [0], [0], [0], [0], [0], [0], [+1, +2]),
        _features('N5dc', [0, 1, 2], [1], [], [], [], [0], [], [0], [0], [+1]),
        _features('N5ddc', [0], [2], [], [], [], [0], [], [0], [0], [+1]),
        _features('N5dddc', [0], [3], [], [], [], [0], [], [0], [0], [-1]),
        _features('N5tc', [0, 1], [0], [0], [0], [0], [1], [], [0], [0], [+1]),
        _features('N5b', [0, 1], [0], [0], [0], [0], [0], [], [2], [0], [0, +1]),
        _features('N5bd', [0, 1], [1], [], [], [], [0], [], [2], [0], [0]),
    ],
    'O': [
        _features('Oa', [0], [0], [], [], [], [0], [0], [0], [3], [0]),
        _features('O0sc', [0, 1], [0], [], [], [], [0], [], [0], [3], [-1]),
        _features('O2s', [0, 1, 2], [0], [], [], [], [0], [], [0], [2], [0]),
        _features('O2sc', [0, 1], [0], [], [], [], [0], [], [0], [2], [+1]),
        _features('O2d', [0], [1], [], [], [], [0], [], [0], [2], [0]),
        _features('O4sc', [0, 1, 2, 3], [0], [], [], [], [0], [], [0], [1], [+1]),
        _features('O4dc', [0, 1], [1], [], [], [], [0], [], [0], [1], [+1]),
        _features('O4tc', [0], [0], [0], [0], [0], [1], [], [0], [1], [+1]),
        _features('O4b', [0], [0], [0], [0], [0], [0], [], [2], [1], [0]),
    ],
    'Si': [
        _features('Sis', [], [0], [], [], [], [0], [], [0], [], []),
        _features('Sid', [], [1], [], [0], [], [0], [], [0], [], []),
        _features('Sidd', [], [2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0], [], [0], [], []),
        _features('Sit', [], [0], [], [], [], [1], [], [0], [], []),
        _features('SiO', [], [1], [], [1], [], [0], [], [0], [], []),
        _features('Sib', [], [0], [], [], [], [0], [], [2], [], []),
        _features('Sibf', [], [0], [], [], [], [0], [], [3], [], []),
        _features('Siq', [0], [0], [0], [0], [0], [0], [1], [0], [], []),
    ],
    'P': [
        _features('P0sc', [0, 1], [0], [0], [0], [0], [0], [], [0], [3], [-2]),
        _features('P1s', [0, 1], [0], [0], [0], [0], [0], [], [0], [2], [0]),
        _features('P1sc', [0, 1, 2], [0], [], [], [], [0], [], [0], [2], [-1]),
        _features('P1dc', [0], [1], [], [], [], [0], [], [0], [2], [-1]),
        _features('P3s', [0, 1, 2, 3], [0], [0], [0], [0], [0], [], [0], [1], [0]),
        _features('P3d', [0, 1], [1], [], [], [], [0], [], [0], [1], [0]),
        _features('P3t', [0], [0], [0], [0], [0], [1], [], [0], [1], [0]),
        _features('P3b', [0], [0], [0], [0], [0], [0], [], [2], [1], [0]),
        _features('P5s', [0, 1, 2, 3, 4, 5], [0], [0], [0], [0], [0], [0], [0], [0], [0]),
        _features('P5sc', [0, 1, 2, 3, 4, 5, 6], [0], [0], [0], [0], [0], [0], [0], [0], [-1, +1, +2]),
        _features('P5d', [0, 1, 2, 3], [1], [], [], [], [0], [], [0], [0], [0]),
        _features('P5dd', [0, 1], [2], [], [], [], [0], [], [0], [0], [0]),
        _features('P5dc', [0, 1, 2], [1], [], [], [], [0], [], [0], [0], [+1]),
        _features('P5ddc', [0], [2], [], [], [], [0], [], [0], [0], [+1]),
        _features('P5t', [0, 1, 2], [0], [], [], [], [1], [], [0], [0], [0]),
        _features('P5td', [0], [1], [], [], [], [1], [], [0], [0], [0]),
        _features('P5tc', [0, 1], [0], [0], [0], [0], [1], [], [0], [0], [+1]),
        _features('P5b', [0, 1], [0], [0], [0], [0], [0], [], [2], [0], [0, +1]),
        _features('P5bd', [0], [1], [], [], [], [0], [], [2], [0], [0]),
    ],
    'S': [
        _features('Sa', [0], [0], [0], [0], [0], [0], [], [0], [3], [0]),
        _features('S0sc', [0, 1], [0], [], [], [], [0], [], [0], [3], [-1]),
        _features('S2s', [0, 1, 2], [0], [0], [0], [0], [0], [], [0], [2], [0]),
        _features('S2sc', [0, 1, 2, 3], [0], [0], [0], [0], [0], [], [0], [2], [-1, +1]),
        _features('S2d', [0], [1], [], [], [], [0], [], [0], [2], [0]),
        _features('S2dc', [0, 1], [1, 2], [], [], [], [0], [], [0], [2], [-1]),
        _features('S2tc', [0], [0], [], [], [], [1], [], [0], [2], [-1]),
        _features('S4s', [0, 1, 2, 3, 4], [0], [0], [0], [0], [0], [], [0], [1], [0]),
        _features('S4sc', [0, 1, 2, 3, 4, 5], [0], [0], [0], [0], [0], [], [0], [1], [-1, +1]),
        _features('S4d', [0, 1, 2], [1], [], [], [], [0], [], [0], [1], [0]),
        _features('S4dd', [0], [2], [], [], [], [0], [], [0], [1], [0]),
        _features('S4dc', [0, 1, 2, 3, 4, 5], [1, 2], [], [], [], [0], [], [0], [1], [-1, +1]),
        _features('S4b', [0], [0], [0], [0], [0], [0], [], [2], [1], [0]),
        _features('S4t', [0, 1], [0], [], [], [], [1], [], [0], [1], [0]),
        _features('S4tdc', [0, 1, 2], [0, 1, 2], [], [], [], [1, 2], [], [0], [1], [-1, +1]),
        _features('S6s', [0, 1, 2, 3, 4, 5, 6], [0], [], [], [], [], [0], [0], [0], [0]),
        _features('S6sc', [0, 1, 2, 3, 4, 5, 6, 7], [0], [], [], [], [], [0], [0], [0], [-1, +1, +2]),
        _features('S6d', [0, 1, 2, 3, 4], [1], [], [], [], [], [0], [], [0], [0]),
        _features('S6dd', [0, 1, 2], [2], [], [], [], [], [0], [], [0], [0]),
        _features('S6ddd', [0], [3], [], [], [], [], [0], [], [0], [0]),
        _features('S6dc', [0, 1, 2, 3, 4, 5], [1, 2, 3], [], [], [], [], [0], [0], [0], [-1, +1, +2]),
        _features('S6t', [0, 1, 2, 3], [0], [], [], [], [1], [], [0], [0], [0]),
        _features('S6td', [0, 1], [1], [], [], [], [1], [], [0], [0], [0]),
        _features('S6tt', [0], [0], [], [], [], [2], [], [0], [0], [0]),
        _features('S6tdc', [0, 1, 2, 3, 4], [0, 1, 2], [], [], [], [1, 2], [], [0], [0], [-1, +1]),
    ],
    'F': [
        _features('F1s', [0, 1], [0], [0], [0], [0], [0], [], [0], [3], [0]),
    ],
    'Cl': [
        _features('Cl1s', [0, 1], [0], [0], [0], [0], [0], [], [0], [3], [0]),
    ],
    'Br': [
        _features('Br1s', [0, 1], [0], [0], [0], [0], [0], [], [0], [3], [0]),
    ],
    'I': [
        _features('I1s', [0, 1], [0], [0], [0], [0], [0], [], [0], [3], [0]),
    ],
    'Li': [
        _features('Li0', [0, 1], [0], [0], [0], [0], [0], [], [0], [0], [0]),
        _features('Li+', [0], [0], [0], [0], [0], [0], [], [0], [0], [+1]),
    ],
}


def _get_atom_features(atom):
    """Compute the RMG feature vector for an RDKit atom (explicit-H graph)."""
    single = all_double = r_double = o_double = s_double = 0
    triple = benzene = quadruple = 0
    bond_order = 0.0
    for bond in atom.GetBonds():
        other_symbol = bond.GetOtherAtom(atom).GetSymbol()
        bond_type = bond.GetBondType()
        if bond_type == rdchem.BondType.SINGLE:
            single += 1
            bond_order += 1
        elif bond_type == rdchem.BondType.DOUBLE:
            all_double += 1
            bond_order += 2
            if other_symbol == 'O':
                o_double += 1
            elif other_symbol == 'S':
                s_double += 1
            else:
                r_double += 1
        elif bond_type == rdchem.BondType.TRIPLE:
            triple += 1
            bond_order += 3
        elif bond_type == rdchem.BondType.AROMATIC:
            benzene += 1
            bond_order += 1.5
        else:
            bond_order += 1

    symbol = atom.GetSymbol()
    unpaired = atom.GetNumRadicalElectrons()
    charge = atom.GetFormalCharge()

    from rmgpu.molecule.adjlist import get_lone_pairs
    lone_pairs = get_lone_pairs(symbol, unpaired, charge, bond_order)

    return {
        'single': single,
        'all_double': all_double,
        'r_double': r_double,
        'o_double': o_double,
        's_double': s_double,
        'triple': triple,
        'quadruple': quadruple,
        'benzene': benzene,
        'lone_pairs': lone_pairs,
        'charge': charge,
    }


def _matches(features, spec):
    """True if atom `features` satisfy atom type `spec` (wildcard = empty list)."""
    for key in _FEATURE_KEYS:
        allowed = spec[key]
        if allowed == []:
            continue  # wildcard
        if features[key] not in allowed:
            return False
    return True


def match_atom_type(symbol, features):
    """
    Return the RMG-Py atom type label for an atom of `symbol` with the given
    feature dict, or the bare element symbol if no specific type matches.
    """
    for spec in _SPECIFIC.get(symbol, []):
        if _matches(features, spec):
            return spec['label']
    return symbol


def assign_atom_types(mol):
    """
    Assign RMG atom types to each atom of the molecule (including hydrogen),
    computed on the graph with explicit hydrogens, matching RMG-Py.

    Returns a list of atom type labels in atom order (heavy atoms first, then
    hydrogens, as produced by RDKit AddHs).
    """
    rdmol = mol._with_explicit_h()
    assignments = []
    for atom in rdmol.GetAtoms():
        symbol = atom.GetSymbol()
        features = _get_atom_features(atom)
        assignments.append(match_atom_type(symbol, features))
    return assignments


def add_atom_types_to_molecule(mol):
    """Assign atom types and store them as a property on the molecule."""
    atom_types = assign_atom_types(mol)
    mol._rdkit.SetProp('atom_types', ','.join(atom_types))
    return atom_types
