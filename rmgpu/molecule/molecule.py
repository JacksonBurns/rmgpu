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
