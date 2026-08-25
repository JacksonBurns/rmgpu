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
    def to_adjlist(self):
        """
        Serialize this molecule to RMG adjacency list format.
        
        Returns a string in RMG adjacency list format.
        """
        from rmgpu.molecule.adjlist import serialize_adjlist
        
        # Add explicit hydrogens for adjacency list format
        from rdkit import Chem
        rdmol_with_h = Chem.AddHs(self._rdkit)
        
        # Create a temporary molecule with explicit Hs
        atoms_info = self._get_atoms_info_from_rdmol(rdmol_with_h)
        n_rad = sum(a['unpaired'] for a in atoms_info)
        multiplicity = n_rad + 1
        
        return serialize_adjlist(
            mol=self,
            multiplicity=multiplicity,
            metal='',
            facet='',
            label='',
            remove_h=False,
            atoms_info=atoms_info
        )

    def _get_atoms_info_from_rdmol(self, rdmol):
        """
        Get atom info dicts from a given RDKit molecule.
        """
        atoms_info = []
        
        # Get ring info
        ri = rdmol.GetRingInfo()
        
        for atom in rdmol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            unpaired = atom.GetNumRadicalElectrons()
            
            # Get lone pairs (approximate using formal charge and valence)
            lone_pairs = 0
            
            charge = atom.GetFormalCharge()
            
            # Get site/morphology from properties if present
            site = atom.GetProp('site') if atom.HasProp('site') else ''
            morphology = atom.GetProp('morphology') if atom.HasProp('morphology') else ''
            
            # Isotope
            isotope = atom.GetIsotope() if atom.GetIsotope() > 0 else -1
            
            # In ring
            in_ring = ri.IsAtomInAnyRing(idx) if hasattr(ri, 'IsAtomInAnyRing') else False
            
            # Bonds
            bonds = {}
            for bond in atom.GetBonds():
                other_atom = bond.GetOtherAtom(atom)
                other_idx = other_atom.GetIdx()
                bond_type = bond.GetBondType()
                # Map RDKit bond types to numeric orders
                bond_order_map = {
                    rdchem.BondType.SINGLE: 1,
                    rdchem.BondType.DOUBLE: 2,
                    rdchem.BondType.TRIPLE: 3,
                    rdchem.BondType.AROMATIC: 1.5
                }
                order = bond_order_map.get(bond_type, 1)
                bonds[other_idx] = order  # 0-based indexing
            
            # Label from property if present
            label = atom.GetProp('label') if atom.HasProp('label') else ''
            
            atoms_info.append({
                'symbol': symbol,
                'unpaired': unpaired,
                'lone_pairs': lone_pairs,
                'charge': charge,
                'site': site,
                'morphology': morphology,
                'isotope': isotope,
                'in_ring': in_ring,
                'bonds': bonds,
                'label': label
            })
        
        return atoms_info

    def get_atoms_info(self):
        """
        Get a list of atom info dicts for adjacency list serialization.
        
        Returns: list of dicts with keys: symbol, unpaired, lone_pairs, charge, 
                 site, morphology, isotope, in_ring, bonds, label
        """
        atoms_info = []
        
        # Get ring info
        ri = self._rdkit.GetRingInfo()
        
        for atom in self._rdkit.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            unpaired = atom.GetNumRadicalElectrons()
            
            # Get lone pairs (approximate using formal charge and valence)
            # RDKit doesn't directly expose lone pairs, so we use implicit H count
            lone_pairs = atom.GetNumExplicitLonePairs() if hasattr(atom, 'GetNumExplicitLonePairs') else 0
            
            charge = atom.GetFormalCharge()
            
            # Get site/morphology from properties if present
            site = atom.GetProp('site') if atom.HasProp('site') else ''
            morphology = atom.GetProp('morphology') if atom.HasProp('morphology') else ''
            
            # Isotope
            isotope = atom.GetIsotope() if atom.GetIsotope() > 0 else -1
            
            # In ring
            in_ring = ri.IsAtomInAnyRing(idx) if hasattr(ri, 'IsAtomInAnyRing') else False
            
            # Bonds
            bonds = {}
            for bond in atom.GetBonds():
                other_idx = bond.GetBeginAtomIdx()
                if other_idx == idx:
                    other_idx = bond.GetEndAtomIdx()
                else:
                    other_idx = bond.GetBeginAtomIdx()
                bond_type = bond.GetBondType()
                # Map RDKit bond types to numeric orders
                bond_order_map = {
                    rdchem.BondType.SINGLE: 1,
                    rdchem.BondType.DOUBLE: 2,
                    rdchem.BondType.TRIPLE: 3,
                    rdchem.BondType.AROMATIC: 1.5
                }
                order = bond_order_map.get(bond_type, 1)
                bonds[other_idx + 1] = order  # 1-based indexing for adjlist
            
            # Label from property if present
            label = self._get_label(atom)
            
            atoms_info.append({
                'symbol': symbol,
                'unpaired': unpaired,
                'lone_pairs': lone_pairs,
                'charge': charge,
                'site': site,
                'morphology': morphology,
                'isotope': isotope,
                'in_ring': in_ring,
                'bonds': bonds,
                'label': label
            })
        
        return atoms_info

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
        except:
            pass  # Some molecules may not kekulize
        
        # Add explicit hydrogens for adjacency list format
        Chem.AddHs(rwmol)
        
        mol = Chem.Mol(rwmol)
        return cls._from_rdmol(mol)
