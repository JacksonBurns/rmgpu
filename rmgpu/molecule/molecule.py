"""
Molecule wrapper over RDKit for the rmgpu package.

Provides construction from SMILES or InChI, molecular formula, charge/radical
count, and structural equality based on canonical form.
"""

from rdkit import Chem
from rdkit.Chem import inchi, rdMolDescriptors
from rdkit import RDLogger

# Suppress RDKit warnings that are handled internally
RDLogger.DisableLog('rdApp.*')


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
        """Return the canonical SMILES string for this molecule."""
        return Chem.MolToSmiles(self._rdkit)

    def to_inchi(self):
        """Return the InChI string for this molecule."""
        return Chem.MolToInchi(self._rdkit)

    def get_formula(self):
        """
        Return the molecular formula as a string, sorted by Hill system
        (C first, H second, then alphabetical).
        """
        return Chem.rdMolDescriptors.CalcMolFormula(self._rdkit)

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
        query = other._rdkit
        return self._rdkit.HasSubstructMatch(query) and query.HasSubstructMatch(self._rdkit)

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
