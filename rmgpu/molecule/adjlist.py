"""
Adjacency list parser/serializer for RMG format.

Ported from RMG-Py (adjlist.py), adapted to rmgpu's RDKit-based Molecule
wrapper. Both parser and serializer operate on graphs with explicit
hydrogen atoms, matching RMG-Py.
"""
import re
import logging

from rdkit import Chem
from rdkit.Chem import rdchem

# Periodic system data, from RMG-Py rmgpy/molecule/element.py (PeriodicSystem).
VALENCE_ELECTRONS = {
    'H+': 0, 'e': 1, 'H': 1, 'He': 2, 'C': 4, 'N': 5, 'O': 6, 'F': 7,
    'Ne': 8, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Br': 7, 'Ar': 8, 'I': 7,
    'X': 4, 'Li': 1,
}

# Elements treated as H-like for lone-pair purposes (RMG-Py update_lone_pairs).
_NO_LONE_PAIRS = {'H', 'Li'}

BOND_ORDERS = {
    'S': 1, 'D': 2, 'T': 3, 'B': 1.5,
}

BOND_SYMBOLS = {
    1: 'S', 2: 'D', 3: 'T', 1.5: 'B',
}


class InvalidAdjacencyListError(Exception):
    """Raised when an adjacency list cannot be parsed."""
    pass


def get_lone_pairs(symbol, unpaired, charge, bond_order):
    """
    Number of lone pairs on an atom, per RMG-Py Molecule.update_lone_pairs():

        lone_pairs = (valence_electrons - radical_electrons - charge - bond_order) / 2

    Hydrogen (and lithium) always have 0.
    """
    if symbol in _NO_LONE_PAIRS:
        return 0
    ve = VALENCE_ELECTRONS.get(symbol)
    if ve is None:
        return 0
    return (ve - unpaired - charge - int(bond_order)) // 2


def get_atoms_info(rdmol):
    """
    Build a list of atom info dicts from an RDKit mol (explicit hydrogens).

    Each dict has keys: label, symbol, unpaired, lone_pairs, charge, site,
    morphology, isotope, in_ring, bonds. bonds maps 0-based neighbor index to
    numeric bond order (1, 2, 3 or 1.5).
    """
    ri = rdmol.GetRingInfo()
    rings = list(ri.AtomRings())
    atoms_in_ring = set()
    for ring in rings:
        atoms_in_ring.update(ring)
    atoms_info = []
    for atom in rdmol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        unpaired = atom.GetNumRadicalElectrons()
        charge = atom.GetFormalCharge()

        bond_order = 0.0
        bonds = {}
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            bond_type = bond.GetBondType()
            bond_order_map = {
                rdchem.BondType.SINGLE: 1,
                rdchem.BondType.DOUBLE: 2,
                rdchem.BondType.TRIPLE: 3,
                rdchem.BondType.AROMATIC: 1.5,
            }
            order = bond_order_map.get(bond_type, 1)
            bonds[other.GetIdx()] = order
            bond_order += order

        atoms_info.append({
            'label': atom.GetProp('label') if atom.HasProp('label') else '',
            'symbol': symbol,
            'unpaired': unpaired,
            'lone_pairs': get_lone_pairs(symbol, unpaired, charge, bond_order),
            'charge': charge,
            'site': atom.GetProp('site') if atom.HasProp('site') else '',
            'morphology': atom.GetProp('morphology') if atom.HasProp('morphology') else '',
            'isotope': atom.GetIsotope() if atom.GetIsotope() > 0 else -1,
            'in_ring': idx in atoms_in_ring,
            'bonds': bonds,
        })
    return atoms_info


def parse_adjlist(text):
    """
    Parse a new-style RMG adjacency list string into a list of atom dicts.

    Column grammar per atom line (molecules, non-group):
        <number> [<label>] <element> u<N> [p<N>] [c<charge>] [s"site"]
        [m"morphology"] [i<N>] [{aid,order} ...]

    Returns: (multiplicity, metal, facet, atoms) where atoms is a list of
    dicts as produced by get_atoms_info() (bonds keyed by 0-based index).
    """
    text = text.strip()
    lines = text.splitlines()
    if not lines:
        raise InvalidAdjacencyListError("Empty adjacency list.")

    multiplicity = None
    metal = ""
    facet = ""

    # Label line
    if len(lines[0].split()) == 1:
        lines.pop(0)  # stored on the molecule, not needed here

    # Multiplicity
    if lines and lines[0].split()[0] == 'multiplicity':
        line = lines.pop(0)
        match = re.match(r'\s*multiplicity\s+(\d+)\s*$', line)
        if not match:
            raise InvalidAdjacencyListError(f"Invalid multiplicity line '{line}'.")
        multiplicity = int(match.group(1))

    # Metal
    if lines and lines[0].split()[0] == 'metal':
        metal = lines.pop(0).split()[1]

    # Facet
    if lines and lines[0].split()[0] == 'facet':
        facet = lines.pop(0).split()[1]

    atoms = []
    bonds = {}

    for line in lines:
        if not line.strip():
            continue

        data = line.split()
        if not data:
            continue

        # Atom index (1-based in the file)
        aid = int(data[0].strip('.')) - 1  # convert to 0-based

        # Label
        label = ""
        index = 1
        if data[1][0] == '*':
            label = data[1]
            index += 1

        # Element
        symbol = data[index]
        if symbol[0] == '[':
            raise InvalidAdjacencyListError(
                "Molecules cannot have atom type lists (group syntax).")
        index += 1

        # Unpaired electrons: u<N>
        up_state = data[index]
        if not up_state.startswith('u'):
            raise InvalidAdjacencyListError(
                f"Expected 'u' prefix for unpaired electrons, got '{up_state}'.")
        unpaired = int(up_state[1:])
        index += 1

        # Lone pairs: p<N> (optional; 0 if absent)
        lone_pairs = 0
        if index < len(data) and data[index].startswith('p'):
            lone_pairs = int(data[index][1:])
            index += 1

        # Charge: c<charge> (optional; 0 if absent)
        charge = 0
        if index < len(data) and data[index].startswith('c'):
            charge = int(data[index][1:])
            index += 1

        # Site: s"..." (optional)
        site = ""
        if index < len(data) and data[index].startswith('s'):
            site = data[index][1:].strip('"')
            index += 1

        # Morphology: m"..." (optional)
        morphology = ""
        if index < len(data) and data[index].startswith('m'):
            morphology = data[index][1:].strip('"')
            index += 1

        # Isotope: i<N> (optional)
        isotope = -1
        if index < len(data) and data[index].startswith('i'):
            isotope = int(data[index][1:])
            index += 1

        atoms.append({
            'label': label,
            'symbol': symbol,
            'unpaired': unpaired,
            'lone_pairs': lone_pairs,
            'charge': charge,
            'site': site,
            'morphology': morphology,
            'isotope': isotope,
            'in_ring': False,
            'bonds': {},
        })

        # Bonds: {aid2,order}
        for datum in data[index:]:
            datum = datum.strip(',')
            aid2, comma, order = datum[1:-1].partition(',')
            if not comma:
                raise InvalidAdjacencyListError(
                    f"Malformed bond descriptor '{datum}' in adjacency list.")
            aid2 = int(aid2) - 1  # convert to 0-based
            order = BOND_ORDERS.get(order, 1)
            bonds.setdefault(aid, {})[aid2] = order

    # Record each bond on both atoms
    for aid, targets in bonds.items():
        for aid2, order in targets.items():
            atoms[aid]['bonds'][aid2] = order

    if multiplicity is None:
        n_rad = sum(a['unpaired'] for a in atoms)
        multiplicity = n_rad + 1

    return multiplicity, metal, facet, atoms


def serialize_adjlist(multiplicity, metal='', facet='', label='', remove_h=False,
                      atoms_info=None):
    """
    Serialize atom info (explicit hydrogens) to RMG adjacency list format,
    matching RMG-Py to_adjacency_list() output exactly:

      - column widths: number width = digits+1, type width = len+1,
        unpaired width = max digit count, label width = len+1 if any label
      - `u<N> p<N> c<C>` are always emitted as separate space-delimited tokens
      - bond lists sorted by atom order
    """
    if atoms_info is None:
        return ""
    if not atoms_info:
        return ""

    adjlist = ''

    if label:
        adjlist += label + '\n'

    if multiplicity != 1 or any(a['unpaired'] for a in atoms_info):
        adjlist += f"multiplicity {multiplicity}\n"

    if metal:
        adjlist += f"metal {metal}\n"

    if facet:
        adjlist += f"facet {facet}\n"

    # Number the atoms (skip unlabeled H if remove_h)
    atom_numbers = {}
    index = 0
    for i, atom in enumerate(atoms_info):
        if remove_h and atom['symbol'] == 'H' and not atom['label']:
            continue
        atom_numbers[i] = str(index + 1)
        index += 1

    if not atom_numbers:
        return ""

    atom_labels = {i: atoms_info[i]['label'] for i in atom_numbers}
    atom_types = {i: atoms_info[i]['symbol'] for i in atom_numbers}
    atom_unpaired = {i: str(atoms_info[i]['unpaired']) for i in atom_numbers}
    atom_lone_pairs = {i: str(atoms_info[i]['lone_pairs']) for i in atom_numbers}
    atom_charge = {
        i: ('+' + str(atoms_info[i]['charge']) if atoms_info[i]['charge'] > 0
            else str(atoms_info[i]['charge']))
        for i in atom_numbers
    }
    atom_site = {i: ('"' + atoms_info[i]['site'] + '"')
                 if atoms_info[i]['site'] else None for i in atom_numbers}
    atom_morphology = {i: ('"' + atoms_info[i]['morphology'] + '"')
                       if atoms_info[i]['morphology'] else None
                       for i in atom_numbers}
    atom_isotope = {i: atoms_info[i]['isotope'] for i in atom_numbers}

    # Field widths (same rules as RMG-Py to_adjacency_list)
    atom_number_width = max([len(s) for s in atom_numbers.values()]) + 1
    atom_label_width = max([len(s) for s in atom_labels.values()], default=0)
    if atom_label_width > 0:
        atom_label_width += 1
    atom_type_width = max([len(s) for s in atom_types.values()]) + 1
    atom_unpaired_width = max([len(s) for s in atom_unpaired.values()], default=0)

    # Assemble the adjacency list
    for i in atom_numbers:
        # Atom number
        adjlist += '{0:<{1:d}}'.format(atom_numbers[i], atom_number_width)
        # Atom label
        adjlist += '{0:<{1:d}}'.format(atom_labels[i], atom_label_width)
        # Atom type
        adjlist += '{0:<{1:d}}'.format(atom_types[i], atom_type_width)
        # Unpaired electrons
        adjlist += 'u{0:<{1:d}}'.format(atom_unpaired[i], atom_unpaired_width)
        # Lone pairs
        adjlist += ' p{0}'.format(atom_lone_pairs[i])
        # Charge
        adjlist += ' c{0}'.format(atom_charge[i])
        # Site
        if atom_site[i]:
            adjlist += ' s{0}'.format(atom_site[i])
        # Morphology
        if atom_morphology[i]:
            adjlist += ' m{0}'.format(atom_morphology[i])
        # Isotope
        if atom_isotope[i] != -1:
            adjlist += ' i{0}'.format(atom_isotope[i])

        # Bonds, sorted by atom order
        for j in sorted(atoms_info[i]['bonds'].keys()):
            if j not in atom_numbers:
                continue
            order = atoms_info[i]['bonds'][j]
            bond_symbol = BOND_SYMBOLS.get(order, str(order))
            adjlist += ' {{{0},{1}}}'.format(atom_numbers[j], bond_symbol)

        adjlist += '\n'

    return adjlist
