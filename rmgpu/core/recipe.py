"""
ReactionRecipe engine (job-05/step-01).

Pure-Python port of RMG-Py's reaction RECIPE machinery (rmgpy/data/kinetics/
family.py: ReactionRecipe, KineticsFamily.apply_recipe, and the atom/bond
bookkeeping in rmgpy/molecule/molecule.py) onto rmgpu's RDKit-based
Molecule (jobs 01-02).

A recipe is a list of labeled bond/electron actions (the RMG "recipe DSL",
custom RMG IP - PLAN.md 4) that, applied to a labeled reactant structure,
turns the reactants into the products. This module holds the ENGINE core:

- ReactionRecipe: parses/holds the action list (from the family.py data
  format or the rmgdb stored form), get_reverse(), apply_forward/
  apply_reverse with RMG's exact validity rules.
- apply_recipe(): the per-labeling product generation (merge reactants,
  apply the recipe, kekulize aromatic products, split products, net-charge
  check, product ordering, the reversible-family relabel swap).

What this step deliberately does NOT port (later job-05 steps):
- template MATCHING (which atom labeling a reactant set matches a family
  template - step 03, rmgpu/molecule/group.py + core/template.py);
- the product-enumeration orchestration (resonance expansion, dedup,
  degeneracy - step 02, generate_reactions);
- the family loader / KineticsFamilies facade (step 04).

What rmgpu/core/recipe.py also does NOT model (out of scope for the
gas-phase port; RMG-Py has it for the surface families, job-12):
- surface sites (element 'X'), van der Waals bonds (order 0), and the
  is_surface_site() / is_bonded_to_surface() checks in RMG's _apply. A
  vdW-bond action on gas-phase atoms is an ActionError, as in RMG.

Validity rules (RMG-Py spec, ported verbatim):
- Bond actions operate on the two atoms carrying the action's labels; a
  label shared by both center atoms must be carried by exactly two atoms.
- CHANGE_BOND: the bond must exist; the resulting order must stay in
  [0, 4]; a benzene (1.5) bond touched by the recipe invalidates the
  "validAromatic" flag.
- FORM_BOND: only single bonds (order 1; 0 = vdW, surface-only); the bond
  must not already exist.
- BREAK_BOND: the bond must exist (order 0 = vdW break, surface-only).
- GAIN/LOSE_RADICAL and GAIN/LOSE_PAIR: per-atom radical/lone-pair
  bookkeeping; a count may not go negative.
- After the recipe, a product with an invalidated aromatic ring is
  kekulized (RMG Molecule.kekulize; the group-template "purely aromatic
  product" branch returns [] - groups are step 03, so a kekulization
  failure here surfaces as KekulizationError).

Graph representation (the one non-trivial porting decision): RMG-Py works
on a graph whose aromatic rings are explicit 1.5-order "benzene" bonds
(Bond.is_benzene); its kekulize (rmgpy/molecule/kekulize.pyx) resolves
those bonds to S/D by DOF analysis. rmgpu's Molecule stores kekulized
(S/D) bonds but RDKit still marks the ring atoms as aromatic, so the
engine (a) re-aromatizes those bonds to 1.5 when it merges the reactants,
and (b) resolves them with RDKit's kekulizer. RDKit's kekulizer returns a
valid Kekule form, which may differ from RMG's exact bond placement
(e.g. 1,3,5- vs 1,4,6-triene). The products are therefore equivalent to
RMG's as molecules (isomorphic), even if a canonical SMILES string can
differ for aromatic products; the step checks and the job-05 gate compare
structure (isomorphism / label fingerprints), not raw SMILES.

Electron bookkeeping on the RDKit graph: radical electrons live on the
RDKit atom (GetNumRadicalElectrons); lone pairs are not RDKit-native, so
atoms carry an 'lp' property (RMG-Py stores lone pairs on its Atom),
re-derived from the stored 'lp' property when present, else from the
neutral RMG formula (valence_electrons - radicals - charge - bond_order) / 2.
"""

from rdkit import Chem
from rdkit.Chem import rdchem
from rmgpu.molecule.molecule import Molecule


class ActionError(Exception):
    """
    Raised when a recipe action cannot be applied (RMG-Py ActionError /
    InvalidActionError; rmgpu does not need the split).
    """


class KekulizationError(Exception):
    """Raised when an aromatic product cannot be kekulized (RMG-Py)."""


# RMG-Py rmgpy/molecule/element.py: PeriodicSystem.valence_electrons
# (outer-shell electrons; the neutral lone-pair formula).
VALENCE_ELECTRONS = {
    'H': 1, 'He': 2, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6,
    'F': 7, 'Ne': 8, 'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6,
    'Cl': 7, 'Ar': 8, 'K': 1, 'Ca': 2, 'Br': 7, 'I': 7,
}

# RMG-Py rmgpy/molecule/element.py: PeriodicSystem.valences (max bonds).
VALENCES = {
    'H': 1, 'He': 0, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 3, 'O': 2,
    'F': 1, 'Ne': 0, 'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 3, 'S': 2,
    'Cl': 1, 'Ar': 0, 'K': 1, 'Ca': 2, 'Br': 1, 'I': 1, 'X': 4,
}

_VALID_ACTIONS = (
    'CHANGE_BOND', 'FORM_BOND', 'BREAK_BOND',
    'GAIN_RADICAL', 'LOSE_RADICAL', 'GAIN_CHARGE', 'LOSE_CHARGE',
    'GAIN_PAIR', 'LOSE_PAIR',
)

_BOND_ACTION_NAMES = ('CHANGE_BOND', 'FORM_BOND', 'BREAK_BOND')

_ORDER_TO_BONDTYPE = {
    0: rdchem.BondType.SINGLE,  # order-0 (vdW) bonds are never materialized
    1: rdchem.BondType.SINGLE,
    2: rdchem.BondType.DOUBLE,
    3: rdchem.BondType.TRIPLE,
    1.5: rdchem.BondType.AROMATIC,
}


def _is_benzene(order):
    """RMG Bond.is_benzene(): a bond of order 1.5 (aromatic)."""
    return order == 1.5


def _bondtype_for(order):
    return _ORDER_TO_BONDTYPE.get(round(float(order), 6),
                                  rdchem.BondType.SINGLE)


def _total_bond_order(atom):
    """
    Total bond order on an atom, per RMG Atom.get_total_bond_order:
    benzene bonds count as 1.5 each, but an atom carrying exactly three
    benzene bonds uses 4/3 per bond (a fully benzene-bonded sp2 carbon
    then has 4.0).
    """
    benzene = 0
    order = 0.0
    for bond in atom.GetBonds():
        if _is_benzene(bond.GetBondTypeAsDouble()):
            benzene += 1
        else:
            order += bond.GetBondTypeAsDouble()
    if benzene == 3:
        order += benzene * 4 / 3.0
    else:
        order += benzene * 3 / 2.0
    return order


def _lone_pairs(atom):
    """
    Lone pairs on an RDKit atom: the stored 'lp' property (set when a
    labeled reactant is built from an RMG adjacency list, whose p-column is
    authoritative), else the neutral RMG formula
    (valence_electrons - radicals - charge - bond_order) / 2, floored at 0.
    """
    if atom.HasProp('lp'):
        return int(atom.GetProp('lp'))
    symbol = atom.GetSymbol()
    ve = VALENCE_ELECTRONS.get(symbol, 4)
    lp = (ve - atom.GetNumRadicalElectrons() - atom.GetFormalCharge()
          - _total_bond_order(atom)) / 2.0
    return max(0, int(lp))


def _update_charge(atom):
    """
    Port of RMG Atom.update_charge (rmgpy/molecule/molecule.py): recompute
    the formal charge from the neutral valence electrons, the total bond
    order, the radical electrons, and the lone pairs:

        charge = valence_electrons - int(bond_order) - radicals - 2*lone_pairs

    RMG calls this after every GAIN_PAIR/LOSE_PAIR action (lone-pair
    increments recompute the charge) and on every product structure in
    apply_recipe (after the split). Surface sites are out of scope here
    (job-12); their charge is 0 in RMG and 'X' atoms never reach this port.
    """
    ve = VALENCE_ELECTRONS.get(atom.GetSymbol(), 4)
    order = _total_bond_order(atom)
    lp = _lone_pairs(atom)
    atom.SetFormalCharge(ve - int(order) - atom.GetNumRadicalElectrons()
                         - 2 * lp)


def _update_piece(mol):
    """
    Port of the product-structure update step in RMG apply_recipe
    (family.py ~1507-1550): RMG calls struct.update_charge() and then
    struct.update() on each split product piece. Ported here are the
    charge / lone-pair parts (RMG Molecule.update_charge,
    Molecule.update_lone_pairs, and the second Molecule.update_charge
    inside update); the atomtype / symmetry / atom-sort parts of RMG's
    update() are rmgpu molecule-layer concerns handled by later job-05
    steps, not by this engine.

    1. recompute each atom's formal charge from its current lone pairs
       (RMG Molecule.update_charge) - the net-charge check below sees the
       updated charges, as in RMG;
    2. recompute each atom's lone pairs from the neutral formula
       (RMG Molecule.update_lone_pairs; H and Li always 0; a
       non-integer value is kept, as in RMG, which only logs an error);
    3. recompute the formal charge again (RMG update -> update_charge).
    """
    for atom in mol.GetAtoms():
        _update_charge(atom)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ('H', 'Li'):
            atom.SetProp('lp', '0')
            continue
        ve = VALENCE_ELECTRONS.get(atom.GetSymbol(), 4)
        order = _total_bond_order(atom)
        lp = (ve - atom.GetNumRadicalElectrons() - atom.GetFormalCharge()
              - int(order)) / 2.0
        atom.SetProp('lp', repr(lp) if lp != int(lp) else str(int(lp)))
    for atom in mol.GetAtoms():
        _update_charge(atom)


def _re_aromatize(rwmol):
    """
    Convert bonds between two RDKit-aromatic atoms to AROMATIC (1.5) bonds,
    rebuilding the RMG-Py "benzene bond" graph from rmgpu's kekulized
    internals. RDKit marks ring atoms as aromatic even though rmgpu's
    Molecule stores explicit S/D bonds; the recipe engine needs the 1.5
    form (Bond.is_benzene) for validAromatic tracking and kekulize.
    """
    aromatic = set()
    for atom in rwmol.GetAtoms():
        if atom.GetIsAromatic():
            aromatic.add(atom.GetIdx())
    for bond in rwmol.GetBonds():
        if bond.GetBondTypeAsDouble() == 1.5:
            continue
        if bond.GetBeginAtomIdx() in aromatic and bond.GetEndAtomIdx() in aromatic:
            bond.SetBondType(rdchem.BondType.AROMATIC)


def _merge_molecules(structures):
    """
    Merge rmgpu Molecules into a single RWMol (RMG Molecule.merge):
    atoms and bonds of every input are copied into one graph. Labels and
    lone-pair properties are preserved.
    """
    merged = None
    for m in structures:
        piece = Chem.RWMol(m._rdkit)
        if merged is None:
            merged = piece
            continue
        offset = merged.GetNumAtoms()
        for atom in piece.GetAtoms():
            new = merged.AddAtom(Chem.Atom(atom))
            for prop in atom.GetPropsAsDict().keys():
                merged.GetAtomWithIdx(new).SetProp(prop, atom.GetProp(prop))
        for bond in piece.GetBonds():
            i1 = bond.GetBeginAtomIdx() + offset
            i2 = bond.GetEndAtomIdx() + offset
            merged.AddBond(i1, i2, bond.GetBondType())
    return merged


def _kekulize_piece(mol):
    """
    Kekulize one product piece (an RDKit RWMol), the port of RMG
    Molecule.kekulize (rmgpy/molecule/kekulize.pyx): resolves the 1.5-order
    benzene bonds of the aromatic rings to explicit S/D. Raises
    KekulizationError if no Kekule form exists (RMG: the final
    update_atomtypes failure).
    """
    if not any(_is_benzene(b.GetBondTypeAsDouble()) for b in mol.GetBonds()):
        return mol  # nothing aromatic to resolve
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        raise KekulizationError(
            'Unable to kekulize product structure:\n' + Chem.MolToSmiles(mol))
    return mol


def _connected_pieces(rwmol):
    """Split an RDKit mol into its connected components (RMG Graph.split)."""
    visited = set()
    pieces = []
    for start in rwmol.GetAtoms():
        i0 = start.GetIdx()
        if i0 in visited:
            continue
        stack = [i0]
        comp = []
        while stack:
            i = stack.pop()
            if i in visited:
                continue
            visited.add(i)
            comp.append(i)
            for nbr in rwmol.GetAtomWithIdx(i).GetNeighbors():
                if nbr.GetIdx() not in visited:
                    stack.append(nbr.GetIdx())
        # old (original mol) -> new (piece mol) index map
        index_map = {old: new for new, old in enumerate(comp)}
        rw = Chem.RWMol()
        for idx in comp:
            rw.AddAtom(Chem.Atom(rwmol.GetAtomWithIdx(idx)))
        for idx in comp:
            for nbr in rwmol.GetAtomWithIdx(idx).GetNeighbors():
                j = nbr.GetIdx()
                if j in comp and j > idx:
                    b = rwmol.GetBondBetweenAtoms(idx, j)
                    rw.AddBond(index_map[idx], index_map[j], b.GetBondType())
        pieces.append(Chem.Mol(rw))
    return pieces


class _ProductPiece:
    """
    One connected component of a product structure (an RDKit mol with a
    net charge and the labels it carries).
    """

    def __init__(self, mol):
        self.mol = mol
        self.labels = {}  # label -> atom index in `mol`
        for i, atom in enumerate(mol.GetAtoms()):
            label = atom.GetProp('label') if atom.HasProp('label') else ''
            if label:
                self.labels[label] = i
        self.net_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())

    def has_label(self, label):
        return label in self.labels

    def label_index(self, label):
        if label not in self.labels:
            raise ValueError('No atom in the structure has the label "%s".'
                             % label)
        return self.labels[label]

    def label_fingerprint(self):
        """
        Order-independent fingerprint of the labeled atoms: for each label,
        (element, radical, charge) and the sorted list of neighbor bond
        orders. Used by the step checks to verify label placement without
        depending on atom ordering.
        """
        fp = {}
        for label, idx in self.labels.items():
            atom = self.mol.GetAtomWithIdx(idx)
            orders = sorted(round(b.GetBondTypeAsDouble(), 6)
                            for b in atom.GetBonds())
            fp[label] = {
                'element': atom.GetSymbol(),
                'radical': int(atom.GetNumRadicalElectrons()),
                'charge': int(atom.GetFormalCharge()),
                'bonds': orders,
            }
        return fp

    def to_molecule(self):
        """Materialize this piece as an rmgpu Molecule (labels preserved)."""
        return Molecule._from_rdmol(Chem.Mol(self.mol))


class ReactionRecipe:
    """
    A list of actions that, when executed on a labeled reactant structure,
    convert it into the products. Ported from RMG-Py ReactionRecipe.

    Actions (RMG recipe DSL; PLAN.md 4):
        CHANGE_BOND  center1, order, center2 - change bond order by `order`
        FORM_BOND    center1, order, center2 - form a bond (order 1)
        BREAK_BOND   center1, order, center2 - break a bond
        GAIN_RADICAL center, n - add n unpaired electrons
        LOSE_RADICAL center, n - remove n unpaired electrons
        GAIN_CHARGE  center, n - add n formal charge
        LOSE_CHARGE  center, n - remove n formal charge
        GAIN_PAIR    center, n - add n lone pairs
        LOSE_PAIR    center, n - remove n lone pairs
    """

    def __init__(self, actions=None):
        self.actions = [list(a) for a in (actions or [])]

    @classmethod
    def from_data(cls, actions):
        """
        Build a recipe from the family data format: either the list of action
        lists (family.py `recipe(actions=[...])`), or the rmgdb
        kinetics_families_table.recipe column (a JSON string holding
        `{'actions': [...]}`). Action names are normalized to upper case;
        each action must be one of the nine valid actions (RMG
        KineticsFamily.load_recipe).
        """
        if isinstance(actions, str):
            import json
            parsed = json.loads(actions)
            actions = parsed.get('actions', parsed) if isinstance(
                parsed, dict) else parsed
        recipe = cls()
        for action in actions:
            action = list(action)
            action[0] = str(action[0]).upper()
            if action[0] not in _VALID_ACTIONS:
                raise ActionError(
                    'Action {} is not a recognized action. Should be one of '
                    '{}'.format(action[0], _VALID_ACTIONS))
            recipe.add_action(action)
        return recipe

    def add_action(self, action):
        """Append an action (name + parameters, RMG format)."""
        self.actions.append(list(action))

    def get_reverse(self):
        """
        The recipe that undoes this one (RMG ReactionRecipe.get_reverse):
        actions played in reverse order with FORM/BREAK swapped, CHANGE
        sign-flipped, and radical/charge/pair gains and losses swapped.
        """
        other = ReactionRecipe()
        for action in reversed(self.actions):
            if action[0] == 'CHANGE_BOND':
                other.add_action(
                    ['CHANGE_BOND', action[1], str(-int(action[2])), action[3]])
            elif action[0] == 'FORM_BOND':
                other.add_action(['BREAK_BOND', action[1], action[2], action[3]])
            elif action[0] == 'BREAK_BOND':
                other.add_action(['FORM_BOND', action[1], action[2], action[3]])
            elif action[0] == 'LOSE_RADICAL':
                other.add_action(['GAIN_RADICAL', action[1], action[2]])
            elif action[0] == 'GAIN_RADICAL':
                other.add_action(['LOSE_RADICAL', action[1], action[2]])
            elif action[0] == 'GAIN_CHARGE':
                other.add_action(['LOSE_CHARGE', action[1], action[2]])
            elif action[0] == 'LOSE_CHARGE':
                other.add_action(['GAIN_CHARGE', action[1], action[2]])
            elif action[0] == 'LOSE_PAIR':
                other.add_action(['GAIN_PAIR', action[1], action[2]])
            elif action[0] == 'GAIN_PAIR':
                other.add_action(['LOSE_PAIR', action[1], action[2]])
        return other

    # ------------------------------------------------------------------ apply

    def _labeled_atoms(self, rwmol, label):
        """
        Atoms of `rwmol` carrying `label` (RMG struct.get_labeled_atoms).
        Raises ValueError if none (RMG does the same).
        """
        atoms = [a for a in rwmol.GetAtoms()
                 if a.HasProp('label') and a.GetProp('label') == label]
        if not atoms:
            raise ValueError(
                'No atom in the structure has the label "%s".' % label)
        return atoms

    def _apply(self, rwmol, forward, unique):
        """
        Apply this recipe to a single (already merged, re-aromatized,
        labeled) structure in place. Port of RMG ReactionRecipe._apply for
        molecule structures (no Group patterns, no surface sites). `unique`
        is accepted for RMG signature compatibility (RMG's molecule path
        ignores it).

        Raises ActionError on an invalid application (the caller converts
        that to "no products") and KekulizationError if the aromatic
        product cannot be kekulized.
        """
        # RMG: struct.props['validAromatic'] = True (reset per application)
        valid_aromatic = True

        for action in self.actions:
            name = action[0]

            if name in _BOND_ACTION_NAMES:
                # RMG resets the connectivity values here; RDKit has none.
                label1, info, label2 = action[1], action[2], action[3]

                if label1 != label2:
                    atoms1 = self._labeled_atoms(rwmol, label1)
                    atoms2 = self._labeled_atoms(rwmol, label2)
                    if not atoms1 or not atoms2:
                        raise ActionError('Invalid atom labels encountered.')
                    atom1, atom2 = atoms1[0], atoms2[0]
                else:
                    atoms = self._labeled_atoms(rwmol, label1)
                    # RMG: "should never have more than two if this action is valid"
                    if len(atoms) > 2:
                        raise ActionError('Invalid atom labels encountered.')
                    atom1, atom2 = atoms
                if atom1 is atom2:
                    raise ActionError('Invalid atom labels encountered.')

                i1, i2 = atom1.GetIdx(), atom2.GetIdx()
                bond = rwmol.GetBondBetweenAtoms(i1, i2)

                if name == 'CHANGE_BOND':
                    info = int(info)
                    if bond is None:
                        # RMG's only exception is a surface site (vdW bond),
                        # which this gas-phase port does not model.
                        raise ActionError(
                            'Attempted to change a nonexistent bond.')
                    if _is_benzene(bond.GetBondTypeAsDouble()):
                        valid_aromatic = False
                    # RMG Bond._change_bond: order += dir*info, then bounds
                    # check ([-0.0001, 4.0001]); the kekulize DOF analysis
                    # resolves 2.5 -> 2 and 0.5 -> 1 for benzene bonds.
                    delta = info if forward else -info
                    new_order = bond.GetBondTypeAsDouble() + delta
                    if new_order < -0.0001 or new_order > 4.0001:
                        raise ActionError(
                            'Unable to change bond: invalid resulting order '
                            '{:r}.'.format(new_order))
                    if _is_benzene(bond.GetBondTypeAsDouble()):
                        new_order = 2 if info > 0 else 1
                    rwmol.RemoveBond(i1, i2)
                    rwmol.AddBond(i1, i2, _bondtype_for(new_order))
                elif ((name == 'FORM_BOND' and forward)
                      or (name == 'BREAK_BOND' and not forward)):
                    # Form a bond between atom1 and atom2
                    if bond is not None:
                        raise ActionError(
                            'Attempted to create an existing bond.')
                    if int(info) not in (1, 0):
                        raise ActionError(
                            'Attempted to create bond of type {!r}'.format(
                                info))
                    if int(info) == 1:
                        rwmol.AddBond(i1, i2, rdchem.BondType.SINGLE)
                    # info == 0 is a vdW (surface) bond: nothing to form in
                    # the gas-phase port.
                elif ((name == 'BREAK_BOND' and forward)
                      or (name == 'FORM_BOND' and not forward)):
                    # Break the bond between atom1 and atom2
                    if bond is None:
                        if int(info) == 0:
                            # RMG: breaking a vdW bond between unconnected
                            # surface atoms happens at split time. Gas-phase:
                            # surface-only, so this is an error.
                            raise ActionError(
                                'Attempted to remove a nonexistent bond.')
                        raise ActionError(
                            'Attempted to remove a nonexistent bond.')
                    rwmol.RemoveBond(i1, i2)

            elif name in ('LOSE_RADICAL', 'GAIN_RADICAL',
                          'LOSE_CHARGE', 'GAIN_CHARGE'):
                # RMG direction logic: the reverse recipe is applied forward
                # (family path) OR the forward recipe is applied in reverse
                # (direct apply_reverse) - either way, forward=False flips
                # the gain/lose direction (family.py _apply, lines 494-501).
                label, change = action[1], int(action[2])
                for atom in self._labeled_atoms(rwmol, label):
                    for _ in range(abs(change)):
                        if ((name == 'GAIN_RADICAL' and forward)
                                or (name == 'LOSE_RADICAL' and not forward)):
                            atom.SetNumRadicalElectrons(
                                atom.GetNumRadicalElectrons() + 1)
                        elif ((name == 'LOSE_RADICAL' and forward)
                                or (name == 'GAIN_RADICAL' and not forward)):
                            if atom.GetNumRadicalElectrons() < 1:
                                raise ActionError(
                                    'Unable to update Atom due to LOSE_RADICAL '
                                    'action: Invalid radical electron set '
                                    '"{}".'.format(
                                        atom.GetNumRadicalElectrons() - 1))
                            atom.SetNumRadicalElectrons(
                                atom.GetNumRadicalElectrons() - 1)
                        elif ((name == 'GAIN_CHARGE' and forward)
                                or (name == 'LOSE_CHARGE' and not forward)):
                            atom.SetFormalCharge(atom.GetFormalCharge() + 1)
                        elif ((name == 'LOSE_CHARGE' and forward)
                                or (name == 'GAIN_CHARGE' and not forward)):
                            atom.SetFormalCharge(atom.GetFormalCharge() - 1)

            elif name in ('LOSE_PAIR', 'GAIN_PAIR'):
                label, change = action[1], int(action[2])
                for atom in self._labeled_atoms(rwmol, label):
                    for _ in range(abs(change)):
                        if ((name == 'GAIN_PAIR' and forward)
                                or (name == 'LOSE_PAIR' and not forward)):
                            atom.SetProp('lp', str(_lone_pairs(atom) + 1))
                            # RMG increment_lone_pairs: lone_pairs += 1 then
                            # update_charge() (recomputes the formal charge).
                            _update_charge(atom)
                        elif ((name == 'LOSE_PAIR' and forward)
                                or (name == 'GAIN_PAIR' and not forward)):
                            lp = _lone_pairs(atom)
                            if lp < 1:
                                raise ActionError(
                                    'Unable to update Atom due to LOSE_PAIR '
                                    'action: Invalid lone electron pairs set '
                                    '"{}".'.format(lp - 1))
                            atom.SetProp('lp', str(lp - 1))
                            # RMG decrement_lone_pairs: lone_pairs -= 1 then
                            # update_charge().
                            _update_charge(atom)
            else:
                raise ActionError(
                    'Unknown action "%s" encountered.' % name)

        if not valid_aromatic:
            _kekulize_piece(rwmol)

    def apply_forward(self, struct, unique=True):
        """Apply the forward recipe to a labeled structure (in place)."""
        self._apply(struct, True, unique)

    def apply_reverse(self, struct, unique=True):
        """Apply the reverse recipe to a labeled structure (in place)."""
        self._apply(struct, False, unique)


# ---------------------------------------------------------------------------
# The reversible-family relabel swap (RMG apply_recipe lines 1392-1473)
# ---------------------------------------------------------------------------

# Families whose label swap RMG hardcodes by name (family.py apply_recipe).
_RELABEL_SWAPS = {
    '1,2_xy_interchange': {'*1': '*4', '*4': '*1'},
    'h_abstraction': {'*1': '*3', '*3': '*1'},
    'f_abstraction': {'*1': '*3', '*3': '*1'},
    'cl_abstraction': {'*1': '*3', '*3': '*1'},
    'br_abstraction': {'*1': '*3', '*3': '*1'},
}

_RELABEL_SPECIAL = (
    'intra_h_migration', 'intra_ene_reaction',
    '6_membered_central_c-c_shift', '1,2_shiftc',
    'intra_r_add_exo_scission', 'intra_substitutions_isomerization',
    'surface_abstraction', 'surface_abstraction_single_vdw',
)


def _relabel_merged(rwmol, family_label, reverse_map):
    """
    Relabel the atoms of a reversible (self-reverse) family's product
    structure, in place, on the MERGED RWMol (RMG apply_recipe lines
    1392-1473) so the products match the reverse template / forbidden
    structures. RMG applies this to the merged structure BEFORE splitting
    it into pieces, so the subsequent piece ordering (by '*1') sees the
    relabeled labels.

    `rwmol` is the merged product RWMol whose atoms carry the forward
    labels. `reverse_map` is the family's reverse map (RMG reverseMap),
    used only when the family has no hardcoded swap.
    """
    label = family_label.lower()

    # Atom currently carrying each label (RMG atom_labels), captured so the
    # swaps reference the original atoms, not re-looked-up labels.
    atom_labels = {}
    for atom in rwmol.GetAtoms():
        if atom.HasProp('label') and atom.GetProp('label') != '':
            atom_labels[atom.GetProp('label')] = atom

    if label in _RELABEL_SWAPS:
        mapping = _RELABEL_SWAPS[label]
    elif label in _RELABEL_SPECIAL:
        mapping = _special_swap(label, atom_labels)
    else:
        # Use the family's reverse map if it has one (RMG line 1470-1472).
        mapping = dict(reverse_map or {})

    for old, new in mapping.items():
        if old in atom_labels:
            atom_labels[old].SetProp('label', new)


def _special_swap(family_label, atom_labels):
    """
    The label mapping for the families RMG handles with bespoke logic
    (family.py apply_recipe lines 1400-1468). `atom_labels` maps each
    current label to its (merged-structure) atom; `len(atom_labels)` is
    the number of labeled atoms, which RMG uses as the chain length for
    intra_H_migration. Surface families raise (support is job-12).
    """
    if family_label == 'intra_h_migration':
        # swap the two ends between which the H moves, then reverse the chain
        mapping = {'*1': '*2', '*2': '*1'}
        highest = len(atom_labels)
        if highest > 4:
            mapping['*4'] = '*5'
            mapping['*5'] = '*4'
        if highest > 6:
            for i in range(6, highest + 1):
                mapping['*{}'.format(i)] = '*{}'.format(6 + highest - i)
        return mapping

    if family_label == 'intra_ene_reaction':
        return {'*1': '*2', '*2': '*1', '*3': '*5', '*5': '*3'}

    if family_label == '6_membered_central_c-c_shift':
        return {'*1': '*3', '*3': '*1', '*4': '*6', '*6': '*4'}

    if family_label == '1,2_shiftc':
        return {'*2': '*3', '*3': '*2'}

    if family_label == 'intra_r_add_exo_scission':
        return {'*1': '*3', '*3': '*1'}

    if family_label == 'intra_substitutions_isomerization':
        return {'*2': '*3', '*3': '*2'}

    if family_label == 'surface_abstraction':
        return {'*1': '*3', '*2': '*5', '*3': '*1', '*5': '*2'}

    if family_label == 'surface_abstraction_single_vdw':
        return {'*1': '*5', '*5': '*1', '*2': '*4', '*4': '*2'}

    raise ValueError(
        'No relabel rule for reversible family "{}".'.format(family_label))


# ---------------------------------------------------------------------------
# apply_recipe
# ---------------------------------------------------------------------------

def apply_recipe(reactant_structures, recipe, family_label='', forward=True,
                 unique=True, relabel_atoms=True, own_reverse=True,
                 reverse_map=None, product_num=None, electrons=0):
    """
    Apply `recipe` to the labeled `reactant_structures` (a list of rmgpu
    Molecule whose atoms already carry the template labels) and return the
    product Molecules in RMG's order, or None if the template is not a match
    (wrong product count / net-charge mismatch).

    Port of RMG-Py KineticsFamily.apply_recipe (family.py 1339-1608) for
    molecule reactants:

    1. merge the reactants into one structure (RMG Molecule.merge; the
       reactants are copied, the originals are untouched);
    2. re-aromatize the ring bonds (RMG benzene-bond graph) and apply the
       recipe (forward recipe, or the reverse recipe applied forward);
    3. kekulize a product with an invalidated aromatic ring (inside _apply);
       for a self-reverse family, relabel the product labels on the merged
       structure (relabel_atoms=True) BEFORE the split, so the '*1'
       ordering below sees the relabeled labels (RMG apply_recipe order);
    4. split the product structure into its pieces (RMG Molecule.split);
    5. check the product count against `product_num` (a mismatch means the
       template is not a match -> None);
    6. check net charge reactants == products (with the family's free
       electrons) -> None on mismatch;
    7. order the products (the piece carrying '*1' first of two; three
       pieces sorted by their lowest label number).

    `product_num` is the template's product count the caller resolves (RMG
    forward: self.product_num or len(forward_template.products); RMG
    reverse: self.reactant_num or len(reverse_template.products)). When
    None the count check is skipped - the caller (the product-enumeration
    orchestration, step 02) must then verify the piece count against the
    template itself. `electrons` is the family's free-electron count (RMG
    self.electrons; 0 for standard gas-phase families). `own_reverse` (RMG
    `not self.reverse_template`): families that are their own reverse
    relabel the products; families with a separate reverse template do not.
    """
    if not reactant_structures:
        raise ValueError('apply_recipe requires at least one reactant.')

    # 1. merge (deep copies; the input molecules are not modified)
    merged = _merge_molecules(reactant_structures)

    # 2. rebuild the RMG benzene-bond graph, then apply the recipe
    _re_aromatize(merged)
    if forward:
        recipe.apply_forward(merged, unique)
    else:
        recipe.apply_reverse(merged, unique)

    # 3. relabel the products of a self-reverse family on the MERGED
    #    structure, BEFORE the split (RMG apply_recipe lines 1392-1473;
    #    own_reverse == RMG `not self.reverse_template`). The relabel must
    #    precede the split + ordering so the '*1' ordering sees the final
    #    labels.
    if relabel_atoms and own_reverse and family_label:
        _relabel_merged(merged, family_label, reverse_map)

    # 4. split into connected pieces
    pieces = [_ProductPiece(mol) for mol in _connected_pieces(merged)]

    # 5. product-count check (template not a match -> None); skipped when
    #    the caller did not supply the template's product count.
    if product_num is not None and product_num != len(pieces):
        return None

    # 5b. update each product structure's lone pairs and charges (RMG
    #     apply_recipe ~1507-1550: struct.update_charge() then
    #     struct.update() on every product piece, BEFORE the net-charge
    #     check, so the check sees the updated charges).
    for piece in pieces:
        _update_piece(piece.mol)
        piece.net_charge = sum(a.GetFormalCharge()
                               for a in piece.mol.GetAtoms())

    # 6. net-charge check (RMG lines 1507-1555)
    reactant_net_charge = sum(m.get_charge() for m in reactant_structures)
    product_net_charge = sum(p.net_charge for p in pieces)

    if electrons < 0:
        if forward:
            reactant_net_charge += electrons
        else:
            product_net_charge += electrons
    elif electrons > 0:
        if forward:
            product_net_charge -= electrons
        else:
            reactant_net_charge -= electrons

    if reactant_net_charge != product_net_charge:
        return None

    # 7. order the products (RMG lines 1557-1571)
    if len(pieces) == 2:
        if not pieces[0].has_label('*1') and pieces[1].has_label('*1'):
            pieces.reverse()
    elif len(pieces) == 3:
        lowest = []
        for piece in pieces:
            nums = [int(l.lstrip('*')) for l in piece.labels.keys()
                    if l.startswith('*') and l.lstrip('*').isdigit()]
            lowest.append(min(nums) if nums else 0)
        pieces = [p for _, p in sorted(zip(lowest, pieces))]

    return [p.to_molecule() for p in pieces]


# ---------------------------------------------------------------------------
# Labeling helpers (used by step-02 product enumeration and the tests)
# ---------------------------------------------------------------------------

def label_fingerprint(molecule):
    """
    Order-independent structural fingerprint of a Molecule's labeled atoms:
    for each label, (element, radical, charge) and the sorted list of
    neighbor bond orders (1.5 for aromatic bonds). Used by the step checks
    to verify label placement without depending on atom ordering.
    """
    fp = {}
    for i, atom in enumerate(molecule._rdkit.GetAtoms()):
        label = atom.GetProp('label') if atom.HasProp('label') else ''
        if not label:
            continue
        orders = sorted(round(b.GetBondTypeAsDouble(), 6) for b in atom.GetBonds())
        fp[label] = {
            'element': atom.GetSymbol(),
            'radical': int(atom.GetNumRadicalElectrons()),
            'charge': int(atom.GetFormalCharge()),
            'bonds': orders,
        }
    return fp


def clear_labeled_atoms(structures):
    """Remove all atom labels (and lp bookkeeping) from `structures`
    (RMG clear_labeled_atoms)."""
    for m in structures:
        for atom in m._rdkit.GetAtoms():
            if atom.HasProp('label'):
                atom.ClearProp('label')
            if atom.HasProp('lp'):
                atom.ClearProp('lp')
    return structures


def label_atoms(structures, maps):
    """
    Tag the atoms of `structures` with the labels of the template atoms they
    matched (RMG _generate_product_structures lines 1627-1631). `maps` is a
    list of dicts (one per structure) mapping reactant atom indices (in
    RDKit order) to the label strings of the template atoms.

    Also, for GAIN_PAIR/LOSE_PAIR bookkeeping: the rmgpu Molecule does not
    keep RMG adjacency-list p-column values, so use
    label_atoms_with_lone_pairs to seed the 'lp' bookkeeping from the
    reactant's p-column; without it the neutral RMG formula is used.
    """
    for structure, mapping in zip(structures, maps):
        for atom_index, label in mapping.items():
            atom = structure._rdkit.GetAtomWithIdx(int(atom_index))
            atom.SetProp('label', label)
    return structures


def label_atoms_with_lone_pairs(structures, maps, lone_pairs):
    """
    As label_atoms, but also stores per-atom lone pairs (the p-column of
    the RMG adjacency list the reactant was built from) as the 'lp'
    property, so GAIN_PAIR/LOSE_PAIR bookkeeping starts from RMG's stored
    value rather than the neutral-formula default.
    """
    for structure, mapping, lp_mapping in zip(structures, maps, lone_pairs):
        for atom_index, label in mapping.items():
            atom = structure._rdkit.GetAtomWithIdx(int(atom_index))
            atom.SetProp('label', label)
            if lp_mapping is not None and int(atom_index) in lp_mapping:
                atom.SetProp('lp', str(int(lp_mapping[int(atom_index)])))
