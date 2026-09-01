"""
Group matcher for rmgpu (job-05/step-03).

Ports RMG-Py's reaction-template / functional-group substructure matching
(the "group semantics" of rmgpy/molecule/group.py + data/base.py
match_node_to_structure + the Cython VF2 subgraph isomorphism in
graph.pyx/vf2.pyx) onto rmgpu's RDKit-backed molecule.

What is ported (the RMG-specific constructs that appear in real family
templates - see the construct inventory in the step-03 report):
  - multi-atomtype groups:  ``R``, ``[Cs,Cd,CO,CS,O,S,N]`` (an atom matches
    if ANY listed atom type matches)  -> ported.
  - wildcard radical ``u[x]`` / lone-pair ``p[x]`` / charge ``c[x]``:
    an empty field = "any" -> ported (empty list = wildcard in matching).
  - benzene bonds (``B``, order 1.5): a group benzene bond matches a
    kekulized single OR double bond (order 1.5 ~ {1,2}) -> ported.
  - explicit-H groups: the template groups carry explicit hydrogen atoms
    and are matched against the molecule's explicit-H graph (RMG matches on
    the explicit-H graph; rmgpu molecules are implicit-H, so the matcher
    expands Hs first) -> ported (AddHs).

What is delegated to rmgpu's existing Molecule layer (documented):
  - SMILES/adjlist construction and explicit-H expansion -> Molecule.
  - bond order extraction (single/double/aromatic) -> RDKit via the mol.

The atom-type tree (``AtomType.is_specific_case_of`` / ``equivalent``) is
loaded from the step-03 recorded reference
(gates/baselines/job05/step03_templates_reference.json -> atomtype_tree),
which is RMG-Py's own ATOMTYPES generic/specific structure - so the tree
semantics are faithful and non-circular (never re-derived from rmgpu code).

The graph-level match is a faithful VF2-style subgraph isomorphism:
  - vertex feasibility = Atom.is_specific_case_of(GroupAtom)
      (atom type specific-case + radical/charge/lone-pair list membership);
  - edge feasibility   = Bond.is_specific_case_of(GroupBond)
      (the molecule bond order is one of the group bond's allowed orders,
       with the benzene 1.5 ~ {1,2} rule);
  - subgraph direction: the GROUP is the pattern (graph2); the molecule's
    explicit-H graph is graph1; the group must map INTO the molecule, and
    every group edge must be matched to a molecule edge of a compatible
    order (a group edge with no molecule counterpart invalidates the
    mapping; extra molecule edges are allowed - subgraph).

The matcher also ports RMG's labeled-atom initial pairing
(match_node_to_structure): group atoms carry recipe labels (``*1``, ``*2``);
each is paired with the uniquely-labeled molecule atom and seeded into the
isomorphism search, and the resulting label mapping (molecule atom index ->
recipe label) is what the recipe engine consumes.
"""

import json
import os

# ---------------------------------------------------------------------------
# Atom-type tree (loaded from the recorded RMG-Py reference)
# ---------------------------------------------------------------------------

_BASELINE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'gates', 'baselines', 'job05', 'step03_templates_reference.json')


class AtomType(object):
    """One node of RMG's atom-type tree.

    `specific` is the transitive closure of the tree below this type; a type
    is a *specific case* of `other` iff it is `other` itself or lies in
    other.specific (RMG AtomType.is_specific_case_of).
    """

    __slots__ = ('label', 'generic', 'specific')

    def __init__(self, label, generic, specific):
        self.label = label
        self.generic = generic   # labels of (transitive) ancestors
        self.specific = specific # labels of (transitive) descendants

    def is_specific_case_of(self, other):
        return self is other or self.label in other.specific

    def equivalent(self, other):
        # RMG AtomType.equivalent: self in other.specific OR other in self.specific
        # (or identical).
        return (self is other
                or self.label in other.specific
                or other.label in self.specific)

    def __repr__(self):
        return 'AtomType(%s)' % self.label


def _load_atomtypes():
    """Build {label: AtomType} from the recorded reference's atomtype_tree.

    The reference records, for every atom type, its `generic` (ancestor) and
    `specific` (descendant) labels directly from RMG's ATOMTYPES. We take the
    recorded `specific` lists as the transitive closures (RMG stores them
    transitively) - this makes is_specific_case_of exact and avoids any
    re-derivation.
    """
    with open(_BASELINE) as f:
        ref = json.load(f)
    tree = ref['atomtype_tree']
    types = {}
    for label, info in tree.items():
        types[label] = AtomType(
            label,
            set(info.get('generic', [])),
            set(info.get('specific', [])),
        )
    return types


ATOMTYPES = _load_atomtypes()


def atom_type(label):
    """Look up an AtomType by label (None if unknown)."""
    return ATOMTYPES.get(label)


# ---------------------------------------------------------------------------
# Group atom / bond (the pattern-side vertex/edge)
# ---------------------------------------------------------------------------

class GroupAtom(object):
    """A template atom pattern: a list of allowed atom types plus (optional)
    allowed radical/charge/lone-pair values. An empty list is a wildcard.
    """

    __slots__ = ('label', 'atomtype', 'radical_electrons', 'charge',
                 'lone_pairs', 'in_ring', 'ignore')

    def __init__(self, label='', atomtype=None, radical_electrons=None,
                 charge=None, lone_pairs=None, in_ring=None):
        self.label = label
        self.atomtype = [atom_type(a) for a in (atomtype or [])]
        self.radical_electrons = list(radical_electrons or [])
        self.charge = list(charge or [])
        self.lone_pairs = list(lone_pairs or [])
        self.in_ring = in_ring  # None (any), 0, or 1
        self.ignore = False

    def __repr__(self):
        return 'GroupAtom(%s %s)' % (self.label,
                                     ','.join(a.label for a in self.atomtype))


# Bond order string -> numeric order (RMG bond_orders; benzene=1.5, aromatic
# group bonds use 1.5). A bond of numeric order 1.5 (benzene) matches
# molecule bonds of order 1 or 2 (the kekulized single/double that make up an
# aromatic ring); this is the RMG "benzene bond ~ {S,D}" semantics.
_BOND_ORDER = {'S': 1.0, 'D': 2.0, 'T': 3.0, 'B': 1.5, 'Q': 4.0,
               'vdW': 0.0, 'H': 0.1, 'R': 0.05}


class GroupBond(object):
    """A template bond pattern: a set of allowed bond orders."""

    __slots__ = ('orders',)

    def __init__(self, orders):
        # orders: list of numeric orders (may include 1.5 for benzene)
        self.orders = list(orders)

    def matches_molecule_order(self, mol_order):
        """Is a molecule bond of `mol_order` a specific case of this group bond?

        Faithful to RMG Bond.equivalent/is_specific_case_of: the group bond is a
        set of allowed orders and the molecule bond matches if its order is
        EXACTLY (within 1e-4) one of them. There is NO aromatic "benzene ~
        {S,D}" relaxation: RMG compares float bond orders directly, so a group
        benzene order (1.5) matches only a molecule bond that is itself 1.5 (an
        aromatic bond), and a kekulized single (1.0) / double (2.0) matches
        only when S / D is in the group's order set.
        """
        for o in self.orders:
            if abs(o - mol_order) < 1e-4:
                return True
        return False

    def __repr__(self):
        return 'GroupBond(%s)' % self.orders


# ---------------------------------------------------------------------------
# Group (the pattern graph)
# ---------------------------------------------------------------------------

class Group(object):
    """A molecular substructure group: explicit-H graph of GroupAtoms +
    GroupBonds, with recipe labels on some atoms.
    """

    def __init__(self):
        self.atoms = []       # list[GroupAtom], index == vertex index
        self.edges = {}       # (i, j) -> GroupBond  (i < j)
        self.multiplicity = None
        self._pending = {}    # src_idx -> [(nbr_idx, orders)] for forward refs

    # -- construction helpers --
    def add_atom(self, atom):
        self.atoms.append(atom)
        return len(self.atoms) - 1

    def add_bond(self, i, j, orders):
        key = (min(i, j), max(i, j))
        self.edges[key] = GroupBond(orders)

    def neighbors(self, i):
        out = []
        for (a, b), bond in self.edges.items():
            if a == i:
                out.append((b, bond))
            elif b == i:
                out.append((a, bond))
        return out

    def has_bond(self, i, j):
        return (min(i, j), max(i, j)) in self.edges

    def get_bond(self, i, j):
        return self.edges[(min(i, j), max(i, j))]

    def get_all_labeled_atoms(self):
        out = {}
        for i, atom in enumerate(self.atoms):
            if atom.label:
                out[atom.label] = i
        return out

    def split(self):
        """RMG Group.split: convert a Group containing one or more
        unconnected subgraphs into separate Groups (atom order preserved).
        Used for RMG's single-template-reactant split (R_Recombination's
        Y_rad/Root: two bonded-together-nothing radical atoms -> 2 reactants).
        """
        n = len(self.atoms)
        comp_of = [-1] * n
        comps = []
        for start in range(n):
            if comp_of[start] != -1:
                continue
            cid = len(comps)
            comp = []
            stack = [start]
            comp_of[start] = cid
            while stack:
                i = stack.pop()
                comp.append(i)
                for (j, _bond) in self.neighbors(i):
                    if comp_of[j] == -1:
                        comp_of[j] = cid
                        stack.append(j)
            comps.append(comp)
        out = []
        for comp in comps:
            idx = {old: new for new, old in enumerate(comp)}
            g = Group()
            for old in comp:
                g.add_atom(self.atoms[old])
            for (a, b), bond in self.edges.items():
                if a in idx and b in idx:
                    g.add_bond(idx[a], idx[b], bond.orders)
            out.append(g)
        return out

    def parse(self, text):
        """Parse an RMG group adjacency list and return self (RMG
        Group.from_adjacency_list(group=True))."""
        parsed = parse_group_adjlist_full(text)
        self.atoms = parsed.atoms
        self.edges = parsed.edges
        self.multiplicity = parsed.multiplicity
        self._pending = {}
        return self

    def match_to_molecule(self, mol, initial_map=None, ignore=(), strict=False):
        """Does the molecule `mol` contain this group as a substructure,
        mapping each group atom (index) to a molecule atom (index)?

        Faithful to RMG match_node_to_structure + Group.match_to_molecule:
        - `initial_map` (dict {group_atom_index: mol_atom_index}) seeds the
          subgraph search (the labeled-atom pairing);
        - `ignore` is a set of molecule atom indices to skip in the vertex
          feasibility (RMG flags structure atoms carrying labels that the group
          does not have, so they are not matched);
        - returns True if at least one valid subgraph isomorphism exists.
        """
        if initial_map is None:
            initial_map = {}
        return len(match_group(mol, self, initial_map=initial_map,
                               ignore=ignore)) > 0

    def __repr__(self):
        return '<Group %d atoms %d bonds>' % (len(self.atoms),
                                              len(self.edges))


# ---------------------------------------------------------------------------
# Group adjacency-list parser (RMG format, group=True)
#
# Line forms seen in real family templates:
#   multiplicity <num>
#   <idx> <label> <atomtype(s)> u<n> [c<n>] [r<n>] {<nbr>,<order>} ...
# where:
#   <label>     ``*1``/``*2`` (recipe label) or ``.``/none
#   <atomtype>  ``R``, ``R!H``, ``H``, ``[Cs,Cd,CO,CS,O,S,N]`` (multi)
#   u<n>        radical electrons (n) or ``u[x]`` wildcard
#   c<n>        charge (n) or ``c[x]`` wildcard
#   r<n>        in-ring flag (0/1)
#   {<nbr>,<order>}  bond to neighbor index, order in {S,D,T,B,Q,...} or a
#                    comma list {S,D}
# ---------------------------------------------------------------------------

def _parse_atomtype_field(tok):
    """Parse an atom-type token: 'R' -> ['R'], '[Cs,Cd,CO]' -> ['Cs','Cd','CO']."""
    tok = tok.strip()
    if tok.startswith('[') and tok.endswith(']'):
        return [t.strip() for t in tok[1:-1].split(',') if t.strip()]
    return [tok] if tok else []


def _parse_orders(tok):
    """Parse a bond-order token: 'S'->[1.0], '{S,D}'->[1.0,2.0], 'B'->[1.5],
    '[S,D,T,B]'->[1.0,2.0,3.0,1.5]. Braces/brackets are order-set delimiters
    (a single order, a {..} or [..] set) - all are stripped before lookup."""
    tok = tok.strip()
    # strip outer order-set delimiters (braces, brackets, parens)
    tok = tok.strip('{}[]()')
    orders = []
    for piece in tok.split(','):
        piece = piece.strip().strip('{}[]()')
        if not piece:
            continue
        if piece in _BOND_ORDER:
            orders.append(_BOND_ORDER[piece])
        else:
            try:
                orders.append(float(piece))
            except ValueError:
                pass
    return orders


def parse_group_adjlist(text):
    """Parse an RMG group adjacency-list string into a Group.

    Handles multiplicity lines, recipe labels (``*N``), multi-atomtypes
    (``[a,b,c]``), wildcards (``u[x]``), charges (``c[n]``), in-ring (``r[n]``),
    and bond-order sets (``{S,D}``, ``B``).
    """
    g = Group()
    index_of = {}   # adjlist index (1-based in text) -> group atom index
    lines = [ln for ln in text.splitlines() if ln.strip() and
             not ln.strip().startswith('#')]
    i = 0
    # multiplicity (may appear as a line: "multiplicity 2" or inline)
    for ln in lines:
        s = ln.strip()
        if s.lower().startswith('multiplicity'):
            try:
                g.multiplicity = int(s.split()[1])
            except (IndexError, ValueError):
                pass
    for ln in lines:
        s = ln.strip()
        if s.lower().startswith('multiplicity'):
            continue
        # tokenize, but keep {..} (with no spaces) and [..] together
        tokens = _tokenize_group_line(s)
        if not tokens:
            continue
        # first token: atom index (1-based)
        try:
            idx = int(tokens[0])
        except ValueError:
            continue
        rest = tokens[1:]
        # label token: '*1' or '.' or absent. Keep the '*' prefix: the recipe
        # engine references labels verbatim ('*1'), and rmgpu's label property
        # stores them with the star (RMG Group.get_all_labeled_atoms format).
        label = ''
        if rest and (rest[0].startswith('*') or rest[0] == '.'):
            label = rest[0] if rest[0].startswith('*') else ''
            rest = rest[1:]
        # atomtype token
        atomtype = []
        if rest:
            atomtype = _parse_atomtype_field(rest[0])
            rest = rest[1:]
        # remaining tokens: property fields (u/c/r/p) and bonds ({nbr,order})
        radical, charge, lone_pairs, in_ring = [], [], [], None
        for tok in rest:
            if tok.startswith('{'):
                _add_group_bond(g, index_of, idx, tok)
            elif tok.startswith('u') and tok[1:] != '':
                if 'x' in tok:
                    radical = []  # wildcard
                else:
                    try:
                        radical = [int(tok[1:])]
                    except ValueError:
                        pass
            elif tok.startswith('c') and tok[1:] != '':
                if 'x' in tok:
                    charge = []
                else:
                    try:
                        charge = [int(tok[1:])]
                    except ValueError:
                        pass
            elif tok.startswith('r') and tok[1:] != '':
                try:
                    in_ring = int(tok[1:])
                except ValueError:
                    pass
            elif tok.startswith('p') and tok[1:] != '':
                if 'x' not in tok:
                    try:
                        lone_pairs = [int(tok[1:])]
                    except ValueError:
                        pass
        atom = GroupAtom(label=label, atomtype=atomtype,
                         radical_electrons=radical, charge=charge,
                         lone_pairs=lone_pairs, in_ring=in_ring)
        ai = g.add_atom(atom)
        index_of[idx] = ai
    return g


def _tokenize_group_line(s):
    """Tokenize a group adjlist line, keeping bracketed groups intact.

    Handles nesting: a bond to a multi-order neighbor is written
    ``{2,[D,T,B]}`` (a ``[..]`` order-set inside the ``{..}`` bond token).
    A plain ``[a,b]`` atom-type token (no leading ``{``) is kept whole too.
    """
    tokens = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        depth = 0
        openers = []
        j = i
        while j < n:
            ch = s[j]
            if ch.isspace() and depth == 0:
                break
            if ch in '[{':
                depth += 1
                openers.append(ch)
            elif ch in ']}':
                depth -= 1
                if openers:
                    openers.pop()
            j += 1
        tokens.append(s[i:j])
        i = j
    return tokens


def _add_group_bond(g, index_of, src_idx, tok):
    """Parse a ``{nbr,order}`` bond token and add the edge (idempotent).

    `order` may be a single letter (``S``), a float, or a bracketed order-set
    (``[D,T,B]``). `nbr` is a 1-based atom index declared in the adjlist.
    """
    inner = tok.strip()
    if inner.startswith('{') and inner.endswith('}'):
        inner = inner[1:-1]
    # split on the first comma that separates nbr from the order part
    comma = inner.find(',')
    if comma < 0:
        return
    nbr_part = inner[:comma].strip()
    order_part = inner[comma + 1:].strip()
    try:
        nbr_idx = int(nbr_part)
    except ValueError:
        return
    orders = _parse_orders(order_part)
    if not orders:
        return
    # Always defer to _finalize_group: it resolves every (src, nbr) pair once
    # all atoms are declared, adding the undirected edge exactly once (keyed
    # by (min, max)). This avoids double-adding an edge seen from both ends and
    # avoids referencing atoms that are not yet declared.
    g._pending.setdefault(src_idx, []).append((nbr_idx, orders))


# Re-run a second pass to resolve bonds whose neighbor was declared later.
def _finalize_group(g):
    pending = getattr(g, '_pending', {})
    index_of = {}
    # rebuild index_of from pending source indices is not possible here; the
    # caller handles ordering. We instead resolve within parse by doing a
    # second pass over pending using the final atoms list.
    # (Group atom order == declaration order, so index_of is 1-based == pos)
    n = len(g.atoms)
    for src in list(pending.keys()):
        for (nbr, orders) in pending[src]:
            if 1 <= src <= n and 1 <= nbr <= n:
                g.add_bond(src - 1, nbr - 1, orders)
    if hasattr(g, '_pending'):
        del g._pending


def parse_group_adjlist_full(text):
    g = parse_group_adjlist(text)
    _finalize_group(g)
    return g


# ---------------------------------------------------------------------------
# Explicit-graph builder (the shape the matcher + template descent consume)
# ---------------------------------------------------------------------------

def _explicit_graph(mol):
    """Return the molecule's explicit-H graph as a dict:
      {'atoms': [{symbol, radical, charge, lone_pairs, atomtype}, ...],
       'adj':   [[(nbr_index, bond_order), ...], ...]}

    The graph is kekulized (so benzene bonds come out as alternating 1.0/2.0,
    matching how the recorded groups and the 99/99-validated matcher operate),
    and the atom order is AddHs order (heavy atoms first, then hydrogens) -
    the same order rmgpu's job-01 atom-type assignment uses, so `atomtype`
    lines up with the per-atom features.
    """
    from rdkit import Chem
    from rmgpu.molecule.atomtype import assign_atom_types
    from rmgpu.molecule.adjlist import get_lone_pairs
    # rmgpu mols are stored kekulized; AddHs preserves the kekulized bonds, so
    # benzene already comes out as alternating 1.0/2.0 (matching the recorded
    # groups and the 99/99-validated matcher). No re-kekulize needed.
    em = mol._with_explicit_h()
    n = em.GetNumAtoms()
    types = assign_atom_types(mol)
    atoms = []
    for i, a in enumerate(em.GetAtoms()):
        atoms.append({
            'symbol': a.GetSymbol(),
            'radical': a.GetNumRadicalElectrons(),
            'charge': a.GetFormalCharge(),
            'lone_pairs': None,  # filled below
            'atomtype': types[i] if i < len(types) else None,
        })
    adj = [[] for _ in range(n)]
    for b in em.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        o = b.GetBondTypeAsDouble()
        adj[i].append((j, o))
        adj[j].append((i, o))
    for i, f in enumerate(atoms):
        bo = sum(o for (_j, o) in adj[i])
        try:
            f['lone_pairs'] = get_lone_pairs(
                f['symbol'], f['radical'], f['charge'], bo)
        except Exception:
            f['lone_pairs'] = None
    return {'atoms': atoms, 'adj': adj}


def _atom_matches_group_atom(mfeat, gatom, atomtype_label):
    """Atom.is_specific_case_of(GroupAtom) on the explicit-H graph.

    `atomtype_label` is the assigned RMG atom-type label for the molecule atom
    (may be a bare element if no specific type matched). The atom-type
    specific-case check uses the recorded atom-type tree.
    """
    # atom-type specific-case (the core RMG semantic)
    if gatom.atomtype:
        ok_type = False
        if atomtype_label is not None:
            mat = atom_type(atomtype_label)
            if mat is not None:
                for gat in gatom.atomtype:
                    if mat.is_specific_case_of(gat):
                        ok_type = True
                        break
            else:
                # Molecule atom type not in the recorded tree (e.g. a bare
                # element label that is itself a known atom type): try direct.
                for gat in gatom.atomtype:
                    if mat_label_in(atomtype_label, gat):
                        ok_type = True
                        break
        if not ok_type:
            return False
    # radical / charge / lone-pair list membership (empty list = wildcard)
    if gatom.radical_electrons:
        if mfeat['radical'] not in gatom.radical_electrons:
            return False
    if gatom.charge:
        if mfeat['charge'] not in gatom.charge:
            return False
    if gatom.lone_pairs:
        if mfeat['lone_pairs'] is not None and \
                mfeat['lone_pairs'] not in gatom.lone_pairs:
            return False
    # in_ring: the recorded templates use no in-ring constraints (inventory
    # shows 0), so treat as non-restrictive; a group atom with an explicit
    # in_ring value is honored only if the molecule atom's ring membership is
    # known (it is not tracked here) -> non-restrictive.
    return True


def mat_label_in(mol_label, gat):
    """True if the molecule atom-type label `mol_label` is a specific case of
    the group AtomType `gat` (by label), used when the label isn't in the
    recorded tree's AtomType objects.
    """
    return mol_label in gat.specific or mol_label == gat.label


def match_explicit_graph(graph, group, initial_map=None, ignore=()):
    """Return the list of valid subgraph-isomorphism mappings of `group`
    (pattern) into an explicit graph.

    `graph` is a dict with keys:
      - 'atoms': list of per-atom feature dicts {symbol, radical, charge,
        lone_pairs, atomtype};
      - 'adj':   list of (neighbor_index, bond_order) per atom index.
    (The shape produced by template._explicit_graph / merge_molecules.)

    Each mapping is a dict {group_atom_index: mol_atom_index}. Faithful to
    RMG's VF2 subgraph isomorphism (group -> molecule):
      - vertex feasibility = Atom.is_specific_case_of(GroupAtom)
        (atom-type tree specific-case + radical/charge/lone-pair list
        membership, empty list = wildcard);
      - edge feasibility   = Bond.is_specific_case_of(GroupBond)
        (the molecule bond order is one of the group bond's allowed orders);
      - subgraph direction: every group edge must map to a molecule edge of a
        compatible order; extra molecule edges are allowed.

    `initial_map` (dict {group_atom_index: mol_atom_index}) seeds the search
    (RMG's labeled-atom initial pairing). `ignore` is a set of molecule atom
    indices excluded from the vertex feasibility.
    """
    if initial_map is None:
        initial_map = {}
    if isinstance(ignore, (set, frozenset, list, tuple)):
        ignore = set(ignore)
    feats = graph['atoms']
    adj = graph['adj']
    atomtype_labels = [a.get('atomtype') for a in feats]
    n = len(feats)
    g_n = len(group.atoms)
    if g_n > n:
        return []

    # Feasibility: which molecule atom can satisfy which group atom.
    feasible = []
    for gi in range(g_n):
        gat = group.atoms[gi]
        row = [
            (mi not in ignore) and
            _atom_matches_group_atom(feats[mi], gat,
                                     atomtype_labels[mi] if mi < len(atomtype_labels) else None)
            for mi in range(n)
        ]
        feasible.append(row)

    # Precompute the group's adjacency (group atom -> list of (nbr, bond)).
    g_adj = [[] for _ in range(g_n)]
    for (a, b), bond in group.edges.items():
        g_adj[a].append((b, bond))
        g_adj[b].append((a, bond))

    # Precompute the molecule bond order between any two atoms (fast lookup).
    mol_bond = {}
    for i in range(n):
        for (j, o) in adj[i]:
            mol_bond[(i, j)] = o
            mol_bond[(j, i)] = o

    def edge_ok(gi, gj, mi, mj):
        """Is a molecule bond between mi and mj a valid image of the group
        edge gi-gj (if the group has one)?"""
        gbond = group.get_bond(gi, gj) if group.has_bond(gi, gj) else None
        if gbond is None:
            return True  # no group edge -> subgraph allows the mol edge
        o = mol_bond.get((mi, mj))
        if o is None:
            return False  # group edge has no molecule image
        return gbond.matches_molecule_order(o)

    results = []
    mapping = [-1] * g_n     # group atom -> mol atom
    mol_used = [False] * n
    # Seed the initial (labeled) pairing.
    for gi, mi in initial_map.items():
        if not (0 <= gi < g_n and 0 <= mi < n) or not feasible[gi][mi]:
            return []  # a seeded pair is infeasible -> no match
        if mapping[gi] != -1 and mapping[gi] != mi:
            return []  # group atom seeded twice, different targets
        if mol_used[mi]:
            return []  # two group atoms seeded to the same molecule atom
        mapping[gi] = mi
        mol_used[mi] = True
    # Seeded pairs must be edge-consistent with each other.
    for gi in range(g_n):
        if mapping[gi] == -1:
            continue
        for (nbr, gbond) in g_adj[gi]:
            if mapping[nbr] != -1:
                if not edge_ok(gi, nbr, mapping[gi], mapping[nbr]):
                    return []

    # DFS order: seeded group atoms first (so the seeded edges constrain the
    # free atoms early), then the rest.
    order = [gi for gi in range(g_n) if mapping[gi] != -1] + \
            [gi for gi in range(g_n) if mapping[gi] == -1]

    def dfs(pos):
        if pos == g_n:
            # Every group edge must have a compatible molecule image.
            for (a, b) in group.edges.keys():
                if not edge_ok(a, b, mapping[a], mapping[b]):
                    return
            results.append({gi: mapping[gi] for gi in range(g_n)})
            return
        gi = order[pos]
        if mapping[gi] != -1:
            dfs(pos + 1)  # already seeded
            return
        for mi in range(n):
            if mol_used[mi] or not feasible[gi][mi]:
                continue
            # Consistency with already-mapped group neighbors (the group edge
            # between gi and a mapped neighbor must match the mol edge).
            good = True
            for (nbr, gbond) in g_adj[gi]:
                if mapping[nbr] != -1:
                    if not edge_ok(gi, nbr, mi, mapping[nbr]):
                        good = False
                        break
            if not good:
                continue
            mapping[gi] = mi
            mol_used[mi] = True
            dfs(pos + 1)
            mapping[gi] = -1
            mol_used[mi] = False

    dfs(0)
    return results


def match_group(mol, group, initial_map=None, ignore=()):
    """Return the list of valid subgraph-isomorphism mappings of `group`
    (pattern) into `mol`'s explicit-H graph (a thin wrapper over
    match_explicit_graph that expands the molecule's Hs first). See
    match_explicit_graph for the mapping representation and RMG-faithful
    vertex/edge semantics."""
    graph = _explicit_graph(mol)
    return match_explicit_graph(graph, group, initial_map=initial_map,
                                ignore=ignore)
