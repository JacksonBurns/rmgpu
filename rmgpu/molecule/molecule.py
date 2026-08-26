"""
Molecule wrapper over RDKit for the rmgpu package.

Provides construction from SMILES or InChI, molecular formula, charge/radical
count, and structural equality based on canonical form.
"""

from rdkit import Chem
from rdkit.Chem import inchi, rdMolDescriptors
from rdkit.Chem import rdchem
from rdkit import RDLogger

# Suppress RDKit warnings that are handled internally
RDLogger.DisableLog('rdApp.*')


def _is_rdkit_element(symbol):
    """
    Return True if `symbol` is a valid element RDKit can create an atom from.
    Used to validate adjacency-list element symbols before RDKit sees them.
    """
    try:
        a = Chem.Atom(symbol)
        return a.GetAtomicNum() > 0
    except Exception:
        return False


# Formula -> canonical SMILES shortcuts, mirrored from RMG-Py
# rmgpy/molecule/translator.py (MOLECULE_LOOKUPS / RADICAL_LOOKUPS).
# These take priority over backend canonicalization and reproduce RMG's
# canonical strings for small, well-known species.
MOLECULE_LOOKUPS = {
    'N2': 'N#N',
    'CH4': 'C',
    'H2O': 'O',
    'C2H6': 'CC',
    'H2': '[H][H]',
    'H2O2': 'OO',
    'C3H8': 'CCC',
    'Ar': '[Ar]',
    'He': '[He]',
    'CH4O': 'CO',
    'CO': '[C-]#[O+]',
    'O2': 'O=O',
    'C': '[C]',
    'H2S': 'S',
    'NH3': 'N',
    'O3': '[O-][O+]=O',
    'Cl2': '[Cl][Cl]',
    'ClH': 'Cl',
    'I2': '[I][I]',
    'HI': 'I',
    'H': 'H+',
    'e': 'e',
}

RADICAL_LOOKUPS = {
    'CH3': '[CH3]',
    'HO': '[OH]',
    'C2H5': 'C[CH2]',
    'O': '[O]',
    'S': '[S]',
    'N': '[N]',
    'HO2': '[O]O',
    'CH': '[CH]',
    'CH2': '[CH2]',
    'H': '[H]',
    'C': '[C]',
    'O2': '[O][O]',
    'S2': '[S][S]',
    'OS': '[S][O]',
    'HS': '[SH]',
    'H2N': '[NH2]',
    'HN': '[NH]',
    'NO': '[N]=O',
    'F': '[F]',
    'Cl': '[Cl]',
    'Br': '[Br]',
    'I': '[I]',
    'CF': '[C]F',
    'CCl': '[C]Cl',
    'CBr': '[C]Br',
    'e': 'e',
}


def _openbabel_canonical(smiles):
    """
    Canonicalize `smiles` with OpenBabel (RMG's canonicalizer for N/S species).
    Returns the canonical SMILES string, or None if OpenBabel is unavailable
    or fails (caller then falls back to RDKit).
    """
    try:
        from openbabel import openbabel
    except ImportError:
        return None
    try:
        ob_conversion = openbabel.OBConversion()
        ob_conversion.SetInFormat('smi')
        ob_conversion.SetOutFormat('can')
        ob_conversion.AddOption('i')  # drop isomer/stereo info
        obmol = openbabel.OBMol()
        if not ob_conversion.ReadString(obmol, smiles):
            return None
        out = ob_conversion.WriteString(obmol).strip()
        return out or None
    except Exception:
        return None


class Molecule:
    """
    Wrapper around an RDKit RWMol for representing chemical species.

    The molecule is stored in canonical kekulized form. Equality and hashing
    are based on the canonical SMILES string, which is stable under atom
    reordering and equivalent structural representations.

    Construction is available from SMILES or InChI. Adjacency-list
    construction will be added in the adjlist step.
    """

    def __init__(self, smiles=None, inchi=None):
        """
        Create a Molecule from a SMILES string or an InChI string.

        If both are provided, InChI takes precedence and a warning is issued
        (matching RMG-Py behavior).

        Raises ValueError if the input cannot be parsed as a valid molecule.
        """
        if inchi and smiles:
            import logging
            logging.warning(
                'Both SMILES and InChI provided for Molecule instantiation; '
                'using InChI and ignoring SMILES.'
            )

        if inchi:
            self._inchi = inchi
            self._build_from_inchi(inchi)
        elif smiles:
            self._smiles = smiles
            self._build_from_smiles(smiles)
        else:
            raise ValueError('Either smiles or inchi must be provided.')
        
        self._rdkit = Chem.Mol(self._rdkit)  # immutable snapshot

    def _build_from_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f'Invalid SMILES string: {smiles}')
        self._rdkit = Chem.RWMol(mol)
        Chem.Kekulize(self._rdkit)
        self._smiles = Chem.MolToSmiles(self._rdkit)  # canonical

    def _build_from_inchi(self, inchi):
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            raise ValueError(f'Invalid InChI string: {inchi}')
        self._rdkit = Chem.RWMol(mol)
        Chem.Kekulize(self._rdkit)
        self._smiles = Chem.MolToSmiles(self._rdkit)

    def to_smiles(self):
        """
        Return the canonical SMILES string.

        Mirrors RMG-Py's translator.to_smiles:
          1. formula-based lookups (small-molecule shortcuts) take priority;
          2. species containing nitrogen or sulfur are canonicalized with
             OpenBabel (RMG's canonical form for such species);
          3. everything else uses RDKit canonical SMILES.
        """
        formula = self.get_formula()
        try:
            if self._has_radical():
                output = RADICAL_LOOKUPS[formula]
            else:
                output = MOLECULE_LOOKUPS[formula]
            return output
        except KeyError:
            pass
        if self._has_nitrogen_or_sulfur():
            rdkit_smi = Chem.MolToSmiles(self._rdkit)
            ob_smi = _openbabel_canonical(rdkit_smi)
            if ob_smi:
                return ob_smi
        return Chem.MolToSmiles(self._rdkit)

    def _has_radical(self):
        """True if any atom carries an unpaired electron."""
        return any(a.GetNumRadicalElectrons() > 0 for a in self._rdkit.GetAtoms())

    def _has_nitrogen_or_sulfur(self):
        """True if the molecule contains N or S (RMG's OpenBabel trigger)."""
        return any(a.GetSymbol() in ('N', 'S') for a in self._rdkit.GetAtoms())

    def to_inchi(self):
        """Return the InChI string for this molecule."""
        return Chem.MolToInchi(self._rdkit)

    def get_formula(self):
        """
        Return the molecular formula as a string, sorted by Hill system
        (C first, H second, then alphabetical).
        """
        mol = self._rdkit
        # Mols built by hand (e.g. symmetry subgraphs) may not have implicit
        # valence computed; sanitize a throwaway copy so formula lookup works.
        try:
            return Chem.rdMolDescriptors.CalcMolFormula(mol)
        except Exception:
            m = Chem.Mol(mol)
            Chem.SanitizeMol(m)
            return Chem.rdMolDescriptors.CalcMolFormula(m)

    def get_charge(self):
        """
        Return the net charge of the molecule as an integer.

        Computed as the sum of formal charges on all atoms.
        """
        return Chem.GetFormalCharge(self._rdkit)

    def get_radical_count(self):
        """
        Return the total number of unpaired electrons (radical count).

        For each atom, the radical count is the number of unpaired electrons.
        This matches the RMG-Py convention.
        """
        total = 0
        for atom in self._rdkit.GetAtoms():
            total += atom.GetNumRadicalElectrons()
        return total

    @property
    def smiles(self):
        """Return the canonical SMILES string."""
        return self.to_smiles()

    @property
    def inchi(self):
        """Return the InChI string."""
        return self.to_inchi()

    def __eq__(self, other):
        if not isinstance(other, Molecule):
            return False
        return self.to_smiles() == other.to_smiles()

    def __hash__(self):
        return hash(self.to_smiles())

    def __repr__(self):
        return f'Molecule(smiles="{self.to_smiles()}")'

    def __str__(self):
        return f'<Molecule "{self.to_smiles()}">'

    @staticmethod
    def _get_label(atom):
        """Return atom label string or empty string if not present."""
        return atom.GetProp('label') if atom.HasProp('label') else ''

    # --- Label semantics ---
    def set_atom_labels(self, labels):
        """
        Set atom labels on this molecule.
        labels: list of label strings, one per atom in RDKit order.
        """
        if len(labels) != self._rdkit.GetNumAtoms():
            raise ValueError('Number of labels must match number of atoms')
        for i, atom in enumerate(self._rdkit.GetAtoms()):
            atom.SetProp('label', labels[i])

    def get_atom_labels(self):
        """Return list of atom labels, in RDKit order."""
        return [self._get_label(atom) for atom in self._rdkit.GetAtoms()]

    def contains_labeled_atom(self, label):
        """Return True if any atom has the given label."""
        return any(self._get_label(atom) == label for atom in self._rdkit.GetAtoms())

    def get_labeled_atoms(self, label):
        """
        Return indices of atoms with the given label.
        Raises ValueError if none found.
        """
        indices = [i for i, atom in enumerate(self._rdkit.GetAtoms()) if self._get_label(atom) == label]
        if not indices:
            raise ValueError(f'No atom in the molecule {self.to_smiles()} has the label "{label}".')
        return indices

    def get_all_labeled_atoms(self):
        """
        Return dict mapping labels to atom indices. If two or more atoms share a label,
        the value is a list of indices.
        """
        labeled = {}
        for i, atom in enumerate(self._rdkit.GetAtoms()):
            label = self._get_label(atom)
            if label:
                if label in labeled:
                    if isinstance(labeled[label], list):
                        labeled[label].append(i)
                    else:
                        labeled[label] = [labeled[label], i]
                else:
                    labeled[label] = i
        return labeled

    def clear_labeled_atoms(self):
        """Remove all atom labels."""
        for atom in self._rdkit.GetAtoms():
            if atom.HasProp('label'):
                atom.ClearProp('label')

    def copy(self, clear_labels=True):
        """
        Return a copy of this molecule. If clear_labels is True, remove labels from the copy.
        The original is unchanged.
        """
        new_mol = Chem.RWMol(self._rdkit)
        if clear_labels:
            for atom in new_mol.GetAtoms():
                if atom.HasProp('label'):
                    atom.ClearProp('label')
        return Molecule._from_rdmol(new_mol)

    # --- Isomorphism and substructure via RDKit ---
    def is_isomorph(self, other):
        """
        Return True if this molecule is isomorphic to other (RDKit).
        """
        if not isinstance(other, Molecule):
            raise TypeError('other must be a Molecule')
        self_with_h = Chem.AddHs(self._rdkit)
        other_with_h = Chem.AddHs(other._rdkit)
        return self_with_h.HasSubstructMatch(other_with_h) and other_with_h.HasSubstructMatch(self_with_h)

    def is_substructure(self, other):
        """
        Return True if other is a substructure of self (RDKit).
        """
        if not isinstance(other, Molecule):
            raise TypeError('other must be a Molecule')
        query = other._rdkit
        return self._rdkit.HasSubstructMatch(query)

    def substructure_match_count(self, other):
        """
        Return the degeneracy (number of matches) of other in self (RDKit).
        """
        if not isinstance(other, Molecule):
            raise TypeError('other must be a Molecule')
        query = other._rdkit
        matches = self._rdkit.GetSubstructMatches(query)
        return len(matches)

    @classmethod
    def _from_rdmol(cls, rdmol):
        """Create a Molecule from an RDKit mol, without kekulizing again."""
        mol = Chem.Mol(rdmol)
        smiles = Chem.MolToSmiles(mol)
        instance = cls.__new__(cls)
        instance._rdkit = mol
        instance._smiles = smiles
        if mol.HasProp('InChI'):
            instance._inchi = mol.GetProp('InChI')
        else:
            instance._inchi = Chem.MolToInchi(mol)
        return instance

    # --- Adjacency list support ---
    def _with_explicit_h(self):
        """
        Return an RDKit mol equal to this molecule with explicit hydrogen
        atoms added (kekulized). This mirrors how RMG-Py represents a molecule:
        its vertex set includes every hydrogen, and atom types, lone pairs,
        symmetry numbers, and adjacency lists are all computed on that graph.
        """
        from rdkit import Chem
        m = Chem.AddHs(self._rdkit)
        return m

    def to_adjlist(self):
        """
        Serialize this molecule to RMG adjacency list format.

        Returns a string in RMG adjacency list format (explicit hydrogens,
        `u{N} p{N} c{N}` columns, matching RMG-Py to_adjacency_list()).
        """
        from rmgpu.molecule.adjlist import serialize_adjlist, get_atoms_info

        rdmol_with_h = self._with_explicit_h()
        atoms_info = get_atoms_info(rdmol_with_h)
        n_rad = sum(a['unpaired'] for a in atoms_info)
        multiplicity = n_rad + 1

        return serialize_adjlist(
            multiplicity=multiplicity,
            metal='',
            facet='',
            label='',
            remove_h=False,
            atoms_info=atoms_info
        )

    def is_cyclic(self):
        """Return True if the molecule contains any rings."""
        ri = self._rdkit.GetRingInfo()
        return ri.NumRings() > 0

    def get_atoms_info(self):
        """
        Get a list of atom info dicts for adjacency list serialization.

        Returns: list of dicts with keys: symbol, unpaired, lone_pairs, charge,
                 site, morphology, isotope, in_ring, bonds, label
        """
        from rmgpu.molecule.adjlist import get_atoms_info
        return get_atoms_info(self._rdkit)

    @classmethod
    def from_adjacency_list(cls, text, saturate_h=False):
        """
        Create a Molecule from an RMG adjacency list string.
        
        Args:
            text: The adjacency list string
            saturate_h: Whether to add explicit hydrogens (for implicit H in old format)
            
        Returns:
            A Molecule instance
            
        Raises:
            InvalidAdjacencyListError: If the adjacency list is invalid
        """
        from rmgpu.molecule.adjlist import parse_adjlist, InvalidAdjacencyListError
        
        try:
            multiplicity, metal, facet, atoms = parse_adjlist(text)
        except InvalidAdjacencyListError:
            raise
        
        # Build RDKit molecule
        rwmol = Chem.RWMol()
        atom_index_map = {}

        # Validate element symbols up front so unknown elements raise a
        # clean InvalidAdjacencyListError instead of a raw RDKit error.
        for atom in atoms:
            if not _is_rdkit_element(atom['symbol']):
                raise InvalidAdjacencyListError(
                    f"Unknown element '{atom['symbol']}' in adjacency list.")

        # Add atoms
        for aid, atom in enumerate(atoms):
            # aid is now 0-based (aid = i)
            rdkit_atom = Chem.Atom(atom['symbol'])
            rdkit_atom.SetFormalCharge(atom['charge'])
            rdkit_atom.SetNumRadicalElectrons(atom['unpaired'])
            
            # Set isotope if present
            if atom['isotope'] > 0:
                rdkit_atom.SetIsotope(atom['isotope'])
            
            # Set properties
            if atom['label']:
                rdkit_atom.SetProp('label', atom['label'])
            if atom['site']:
                rdkit_atom.SetProp('site', atom['site'])
            if atom['morphology']:
                rdkit_atom.SetProp('morphology', atom['morphology'])
            
            idx = rwmol.AddAtom(rdkit_atom)
            atom_index_map[aid] = idx
        
        # Add bonds (skip duplicates)
        for aid, atom in enumerate(atoms):
            for target_aid, order in atom['bonds'].items():
                if target_aid not in atom_index_map:
                    raise InvalidAdjacencyListError(f"Bond to unknown atom {target_aid}")
                # Skip if already added (avoid duplicates) - aid < target_aid
                if aid >= target_aid:
                    continue
                
                bond_type_map = {
                    1: rdchem.BondType.SINGLE,
                    2: rdchem.BondType.DOUBLE,
                    3: rdchem.BondType.TRIPLE,
                    1.5: rdchem.BondType.AROMATIC
                }
                bond_type = bond_type_map.get(order, rdchem.BondType.SINGLE)
                
                rwmol.AddBond(atom_index_map[aid], atom_index_map[target_aid], bond_type)
        
        # Kekulize if needed
        try:
            Chem.Kekulize(rwmol)
        except Exception:
            pass  # Some molecules may not kekulize

        mol = Chem.Mol(rwmol)
        return cls._from_rdmol(mol)
