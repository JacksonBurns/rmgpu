"""
Template matching (job-05/step-03).

`match(family, reaction) -> template_labels` decides whether a family's
templates describe a concrete reaction (reactants + products), and if so
returns the most-specific template labels - the two things the core loop
(job 06) needs to know which family produced a reaction and with what
kinetics template.

Faithful to the RMG-Py flow (rmgpy/data/kinetics/family.py
get_labeled_reactants_and_products + rmgpy/data/kinetics/groups.py
get_reaction_template):

  1. Reactant -> template matching (the verdict). Each reactant is matched to
     the family's forward-template reactant slot with the group matcher
     (this step's rmgpu/molecule/group.py - 99/99 subgraph-isomorphism parity
     with RMG-Py). The bimolecular A+B / B+A swap branches are tried, and for
     the single-group split (R_Recombination) the two reactants merge into one
     structure (RMG Molecule.merge) and the one group is matched to it.
  2. Recipe + product check. Each matching labeling is applied through the
     step-01 recipe engine (apply_recipe); the family MATCHES only if some
     labeling reproduces the reaction's products (RMG
     get_labeled_reactants_and_products). This is what rejects a WRONG family:
     the same reactants often subgraph-match several families' templates, but
     only the generating family's recipe reproduces the products (the recorded
     verdict matrix - 46 own-family matches, 0 cross-family).
  3. Tree descent (the labels). For a valid labeling, each deduped
     forward-template top node is descended to its most-specific node (RMG
     get_reaction_template walks the deduped top nodes, matching each against
     the reactant that carries its labels); those node labels are the
     reaction's template labels.

The group tree (entries + parent/children) and the atom-type tree are loaded
from the recorded reference (gates/baselines/job05/step03_templates_reference
.json) - RMG-Py's own trees - so the port is faithful and non-circular. The
step-04 family loader (rmgpu/core/family.py) reads the live RMG-database
files into this same TemplateFamily shape.
"""

from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.group import (
    Group,
    atom_type,
    match_explicit_graph,
    _explicit_graph,
)
from rmgpu.core.recipe import (
    ActionError,
    KekulizationError,
    ReactionRecipe,
    apply_recipe,
)
from rmgpu.core import enumeration as enum


# ---------------------------------------------------------------------------
# Reaction container
# ---------------------------------------------------------------------------

class Reaction(object):
    """A reaction: reactant Molecule list + product Molecule list (RMG
    Reaction's molecule subset). Job 06 can pass its own object with
    `.reactants` / `.products`; this is the canonical container."""

    def __init__(self, reactants, products=None):
        self.reactants = list(reactants)
        self.products = list(products) if products is not None else []

    def __repr__(self):
        return '<Reaction %r -> %r>' % (
            [m.get_formula() for m in self.reactants],
            [m.get_formula() for m in self.products])


# ---------------------------------------------------------------------------
# Template group tree (loaded from the reference)
# ---------------------------------------------------------------------------

class TemplateEntry(object):
    """One node of a family's group tree. `item` is a `Group` (concrete
    pattern) or a `LogicNode` (a named OR/AND of component entries);
    `children`/`parent` are TemplateEntry objects."""

    __slots__ = ('label', 'item', 'parent', 'children')

    def __init__(self, label, item, parent=None):
        self.label = label
        self.item = item
        self.parent = parent
        self.children = []

    @property
    def is_logic(self):
        return isinstance(self.item, LogicNode)

    def __repr__(self):
        return '<TemplateEntry %s>' % self.label


class LogicNode(object):
    """A minimal RMG LogicNode: a set of component entry labels (OR/AND),
    optionally inverted. Ported just far enough for template matching."""

    __slots__ = ('symbol', 'components', 'invert')

    def __init__(self, symbol, components, invert=False):
        self.symbol = symbol
        self.components = list(components)  # entry labels (strings)
        self.invert = invert

    def __repr__(self):
        return '<LogicNode %s{%s}>' % (
            self.symbol, ','.join(self.components))


def _make_item(node):
    if node['type'] == 'logic':
        return LogicNode(node.get('symbol') or 'OR',
                         node.get('components') or [],
                         bool(node.get('invert', False)))
    return Group().parse(node['adj'])


class TemplateFamily(object):
    """A family's templates + group tree, loaded from the step-03 reference
    (the same shape the step-04 family loader will emit).

    - `entries`: {label: TemplateEntry} (the full tree, parent/children wired);
    - `top`: list of TemplateEntry (the forward-template top nodes, in order,
      duplicates retained - R_Recombination is [Y_rad, Y_rad]);
    - `forward_template`: list of entry labels, one per template reactant SLOT
      before any single-group split;
    - `reactant_num`: the effective reactant count AFTER the single-group
      split (R_Recombination: 2);
    - `recipe` / `reverse_recipe` / `own_reverse` / `reversible` /
      `allow_charged_species` / `electrons` / `product_num_forward` /
      `reverse_map`: the recipe-application data (step-02 reference / the
      step-04 loader).
    """

    def __init__(self, label, entries, top, forward_template,
                 reactant_num=None, recipe=None, reverse_recipe=None,
                 own_reverse=False, reversible=False,
                 allow_charged_species=False, electrons=0,
                 product_num_forward=None, reverse_map=None):
        self.label = label
        self.entries = entries
        self.top = list(top)
        self.forward_template = list(forward_template)
        if reactant_num is None:
            reactant_num = len(forward_template)
        self.reactant_num = reactant_num
        self.recipe = recipe
        self.reverse_recipe = reverse_recipe
        self.own_reverse = own_reverse
        self.reversible = reversible
        self.allow_charged_species = allow_charged_species
        self.electrons = electrons
        self.product_num_forward = product_num_forward
        self.reverse_map = reverse_map

    @classmethod
    def from_reference(cls, fam_ref):
        """Build the tree from a recorded family dict (the reference
        `families` entry: {label, top, tree, forward_template, ...}). The
        recipe is NOT in the step-03 reference; attach it via
        from_references / direct attribute set."""
        tree = fam_ref['tree']
        entries = {}
        for label, node in tree.items():
            entries[label] = TemplateEntry(label, _make_item(node['node']))
        for label, node in tree.items():
            e = entries[label]
            parent_label = node.get('parent')
            e.parent = entries.get(parent_label) if parent_label else None
            e.children = [entries[c] for c in node.get('children', [])]
        top = [entries[t] for t in fam_ref['top']]
        fwd = fam_ref.get('forward_template', fam_ref['top'])
        own_reverse = bool(fam_ref.get('own_reverse', False))
        reactant_num = len(fwd)
        if len(fwd) == 1:
            item = entries[fwd[0]].item
            if isinstance(item, Group):
                n_labels = len(item.get_all_labeled_atoms())
                if n_labels >= 2:
                    reactant_num = n_labels
        return cls(fam_ref['label'], entries, top, fwd,
                   reactant_num=reactant_num, own_reverse=own_reverse)

    @classmethod
    def from_references(cls, fam_ref, step02_case):
        """Build the full TemplateFamily (tree from the step-03 reference +
        recipe/flags from the step-02 reference's family_meta) - the shape
        `match` consumes. `step02_case` is a step-02 reference case dict.

        `reactant_num` is taken from the step-02 `family_meta`
        (`num_template_reactants_effective`) - RMG's own effective reactant
        count AFTER the single-group split (R_Recombination: 2) - not
        re-derived from label counts (a bare-`*` group like R_Recombination's
        Y_rad has 1 distinct label for 2 reactant radicals, so label counting
        is unreliable)."""
        fam = cls.from_reference(fam_ref)
        meta = step02_case['family_meta']
        fam.recipe = ReactionRecipe.from_data(meta['recipe'])
        if meta.get('reverse_recipe') is not None:
            fam.reverse_recipe = ReactionRecipe.from_data(
                meta['reverse_recipe'])
        fam.own_reverse = meta['own_reverse']
        fam.reversible = meta['reversible']
        fam.allow_charged_species = meta['allow_charged_species']
        fam.electrons = meta['electrons']
        fam.product_num_forward = meta.get('effective_product_num_forward')
        fam.reverse_map = meta.get('reverse_map')
        eff = meta.get('num_template_reactants_effective',
                       meta.get('num_template_reactants'))
        if eff is not None:
            fam.reactant_num = eff
        return fam


# ---------------------------------------------------------------------------
# Component resolution + merge
# ---------------------------------------------------------------------------

def _logic_component_groups(family, entry):
    """The concrete Groups under a LogicNode entry (component entry labels
    resolved through family.entries, recursing into nested LogicNodes)."""
    groups = []
    for comp in entry.item.components:
        e = family.entries.get(comp)
        if e is None:
            continue
        if isinstance(e.item, Group):
            groups.append(e.item)
        elif isinstance(e.item, LogicNode):
            groups.extend(_logic_component_groups(family, e))
    return groups


def _slot_groups(family, slot_index):
    """The concrete Groups of a forward-template slot (its item if a Group,
    or all component Groups for a LogicNode)."""
    entry = family.top[slot_index]
    if isinstance(entry.item, Group):
        return [entry.item]
    return _logic_component_groups(family, entry)


def _dedup_top(fam):
    """The forward-template top entries with duplicates removed in order
    (RMG get_reaction_template dedups; R_Recombination's [Y_rad, Y_rad] ->
    [Y_rad])."""
    seen = []
    for e in fam.top:
        if e not in seen:
            seen.append(e)
    return seen


def merge_graph(mols):
    """Merge rmgpu Molecules into one explicit graph (RMG Molecule.merge).
    Hs are added per molecule (AddHs order); the graphs are concatenated with
    no cross-molecule bonds (reactant merging introduces none). Returns a dict
    with the same shape as group._explicit_graph (atoms + adj)."""
    if len(mols) == 1:
        return _explicit_graph(mols[0])
    per = [_explicit_graph(m) for m in mols]
    atoms = []
    for g in per:
        atoms.extend(g['atoms'])
    adj = [[] for _ in range(len(atoms))]
    offset = 0
    for g in per:
        for i, nbrs in enumerate(g['adj']):
            for (j, o) in nbrs:
                adj[offset + i].append((offset + j, o))
        offset += len(g['atoms'])
    return {'atoms': atoms, 'adj': adj}


# ---------------------------------------------------------------------------
# The matcher (group -> molecule subgraph; the step-02 engine hook)
# ---------------------------------------------------------------------------

class TemplateMatcher(object):
    """The real group matcher, driven off a TemplateFamily's trees.

    Implements the step-02 enumeration matcher interface
    `match_molecule(form, slot, branch)` so it can replace the RecordedMatcher
    in generate_reactions, plus the lower-level `match_graph` used by `match`.
    """

    def __init__(self, family):
        self.family = family

    def match_graph(self, graph, slot_index):
        """Match an explicit `graph` to the slot's template group (any
        component for a LogicNode slot). Returns the list of
        {group_atom_index: graph_atom_index} mappings (one per valid subgraph
        isomorphism, summed across OR components - RMG
        find_subgraph_isomorphisms semantics)."""
        out = []
        for g in _slot_groups(self.family, slot_index):
            for m in match_explicit_graph(graph, g):
                out.append(m)
        return out

    def match_molecule(self, form, slot, branch='ab'):
        """Step-02 engine interface: the list of labelings (dicts
        {molecule_atom_index: recipe_label}) of `form` against template
        reactant `slot` for reactant order `branch`.

        `branch` selects the effective slot for the bimolecular swap:
          - 'ab': template order (A+B) - reactant i -> slot i;
          - 'ba': swapped (B+A) - reactant i -> slot (1 - i).
        For a unimolecular form, slot is 0 and branch is ignored.
        """
        graph = _explicit_graph(form)
        fam = self.family
        eff_slot = (1 - slot) if (branch == 'ba' and len(fam.top) == 2) else slot
        labelings = []
        for g in _slot_groups(fam, eff_slot):
            labelings.extend(
                self._labelings(g, match_explicit_graph(graph, g)))
        return labelings

    @staticmethod
    def _labelings(group, mappings):
        """Invert subgraph mappings {group_atom_index: mol_index} to
        {mol_index: recipe_label} labelings (one per mapping). Uses each group
        atom's OWN label (GroupAtom.label), not the collapsed
        get_all_labeled_atoms dict, so groups where several atoms share one
        label (R_Recombination's Y_rad, both atoms `*`) keep every atom
        labeled."""
        out = []
        for m in mappings:
            lab = {}
            for gi, mi in m.items():
                lbl = group.atoms[gi].label
                if lbl:
                    lab[mi] = lbl
            out.append(lab)
        return out


# ---------------------------------------------------------------------------
# match_node_to_structure + descend_tree (RMG Database / KineticsGroups)
# ---------------------------------------------------------------------------

def _atom_specific_case(group_atom, mol_atom_feat):
    """Atom.is_specific_case_of(GroupAtom) for a single atom (RMG
    match_node_to_structure semantic check #1): atom-type specific-case (the
    recorded atom-type tree) + radical/charge/lone-pair list membership."""
    if group_atom.atomtype:
        mol_type = (atom_type(mol_atom_feat.get('atomtype'))
                    if mol_atom_feat.get('atomtype') else None)
        ok = False
        if mol_type is not None:
            for gat in group_atom.atomtype:
                if mol_type.is_specific_case_of(gat):
                    ok = True
                    break
        if not ok:
            return False
    if group_atom.radical_electrons:
        if mol_atom_feat['radical'] not in group_atom.radical_electrons:
            return False
    if group_atom.charge:
        if mol_atom_feat['charge'] not in group_atom.charge:
            return False
    if group_atom.lone_pairs:
        lp = mol_atom_feat.get('lone_pairs')
        if lp is not None and lp not in group_atom.lone_pairs:
            return False
    return True


def _group_matches(graph, group, labeled, strict=False):
    """Match a concrete Group to an explicit graph with RMG's labeled-atom
    pairing. `labeled` is {recipe_label: graph_atom_index} (the structure's
    labeled atoms). Returns one valid {group_atom_index: graph_index} mapping
    (the initial pairing seeded) or None. Faithful to RMG
    Database.match_node_to_structure (the Group branch): pair each labeled
    group atom with the same-labeled structure atom (strict -> every group
    label must be present), run the atom specific-case check, flag structure
    labels the group lacks as `ignore`, and run the subgraph isomorphism with
    the initial pairing."""
    centers = group.get_all_labeled_atoms()  # {label: group_atom_index}
    initial_map = {}
    for label, center_idx in centers.items():
        if label not in labeled:
            if strict:
                return None
            continue
        atom_idx = labeled[label]
        if not _atom_specific_case(group.atoms[center_idx],
                                   graph['atoms'][atom_idx]):
            return None
        initial_map[center_idx] = atom_idx
    ignore = set(i for lbl, i in labeled.items() if lbl not in centers)
    mappings = match_explicit_graph(graph, group, initial_map=initial_map,
                                    ignore=ignore)
    return mappings[0] if mappings else None


def _node_matches(family, node, graph, labeled, strict=False):
    """RMG Database.match_node_to_structure (the node dispatch): a LogicNode
    matches when any component Group matches (LogicOr.match_to_structure); a
    Group node via _group_matches. Returns one mapping or None."""
    if isinstance(node.item, LogicNode):
        for g in _logic_component_groups(family, node):
            m = _group_matches(graph, g, labeled, strict)
            if m is not None:
                return m
        return None
    return _group_matches(graph, node.item, labeled, strict)


def descend_tree(family, graph, labeled, root=None, strict=True):
    """RMG Database.descend_tree, ported. Descend from a matched top node to
    the most-specific node whose group matches the labeled structure; returns
    the matched TemplateEntry, or None if the root (or the first matching top
    node) does not match. Faithful to RMG: a node is the best match when it
    has exactly one matching child (recurse), no matching children (the node
    itself, or an `Others-*` last child if present), or several matching
    children (the first - RMG's documented tie-break)."""
    if root is None:
        for root in family.top:
            if _node_matches(family, root, graph, labeled, strict):
                break
        else:
            return None
    elif not _node_matches(family, root, graph, labeled, strict):
        return None
    next_node = [c for c in root.children
                 if _node_matches(family, c, graph, labeled, strict)]
    if len(next_node) == 1:
        return descend_tree(family, graph, labeled, next_node[0], strict)
    elif not next_node:
        if root.children and root.children[-1].label.startswith('Others-'):
            return root.children[-1]
        return root
    return descend_tree(family, graph, labeled, next_node[0], strict)


def get_reaction_template(family, graphs, labeled_per):
    """RMG KineticsGroups.get_reaction_template, ported. Descend each deduped
    forward-template top node to its most-specific node and return the list of
    matched node LABELS (the reaction's template labels), or None if any top
    node fails to descend.

    `graphs` is a list of explicit-graph dicts (one per reactant, or the single
    merged graph for the single-group split) and `labeled_per` is a parallel
    list of {recipe_label: graph_atom_index} dicts (the labels applied by the
    valid labeling). Each top node is matched against the reactant that carries
    its labels (a top node whose labels are absent from a reactant does not
    match it). The 7 in-scope families never have a top node spanning two
    reactants, so a top node's labels live in exactly one reactant.
    """
    out = []
    for entry in _dedup_top(family):
        node = None
        for i in range(len(graphs)):
            n = descend_tree(family, graphs[i], labeled_per[i],
                             root=entry, strict=True)
            if n is not None:
                node = n
                break
        if node is None:
            return None
        out.append(node.label)
    return out


# ---------------------------------------------------------------------------
# match(family, reaction) - the deliverable entry point
# ---------------------------------------------------------------------------

def match(family, reaction):
    """Determine whether `family`'s templates describe `reaction`, and if so
    return the most-specific template labels; otherwise None.

    Faithful to RMG's get_labeled_reactants_and_products (reactant->template
    match + recipe + product check) and get_reaction_template_labels (the
    tree descent). A family matches iff SOME valid labeling of its template
    reproduces the reaction's products; the labels are the most-specific node
    labels for that labeling.

    Args:
        family: a TemplateFamily (group tree + recipe + flags).
        reaction: a Reaction (reactants + products as Molecule lists) or any
            object with `.reactants` / `.products`.
    Returns:
        list of template labels (the reaction's template) or None.
    """
    reactants = list(reaction.reactants)
    products = list(reaction.products)
    fam = family
    if fam.recipe is None:
        raise ValueError(
            'match() requires family.recipe to be set (attach the step-02 '
            'family_meta via TemplateFamily.from_references).')
    if len(reactants) != fam.reactant_num:
        return None  # wrong reactant count for this family

    matcher = TemplateMatcher(fam)

    # --- single-group split (R_Recombination): merge reactants, match the
    # one group to the merged graph. ---
    if len(_dedup_top(fam)) == 1 and fam.reactant_num > 1:
        top = _dedup_top(fam)[0]
        if not isinstance(top.item, Group):
            return None  # single-slot LogicNode with >1 reactants: out of scope
        graph = merge_graph(reactants)
        offsets = _reactant_offsets(reactants)
        for m in match_explicit_graph(graph, top.item):
            lab = _merged_labels(top.item, m, reactants, offsets)
            if lab is None:
                continue
            if _apply_and_check(fam, reactants, lab, products):
                # Descent labels: the merged graph + merged labels (the
                # per-molecule label dicts offset into the merged index space).
                merged_lab = {}
                for i, mol_lab in enumerate(lab):
                    for idx, lbl in mol_lab.items():
                        merged_lab[lbl] = idx + offsets[i]
                return get_reaction_template(fam, [graph], [merged_lab])
        return None

    # --- unimolecular (one reactant, single template slot). ---
    if len(reactants) == 1 and len(_dedup_top(fam)) >= 1:
        graph = _explicit_graph(reactants[0])
        for g in _slot_groups(fam, 0):
            for m in match_explicit_graph(graph, g):
                lab = TemplateMatcher._labelings(g, [m])
                if not lab:
                    continue
                lab_mol = lab[0]
                if _apply_and_check(fam, reactants, [lab_mol], products):
                    labeled = {lbl: idx for idx, lbl in lab_mol.items()}
                    return get_reaction_template(fam, [graph], [labeled])
        return None

    # --- bimolecular (two reactants, template order + swapped). ---
    if len(reactants) == 2 and len(_dedup_top(fam)) == 2:
        graphs = [_explicit_graph(m) for m in reactants]
        for branch in ('ab', 'ba'):
            la = matcher.match_molecule(reactants[0], 0, branch)
            lb = matcher.match_molecule(reactants[1], 1, branch)
            if not la or not lb:
                continue
            for map_a in la:
                for map_b in lb:
                    if _apply_and_check(fam, reactants,
                                        [map_a, map_b], products):
                        labeled_per = [
                            {lbl: idx for idx, lbl in map_a.items()},
                            {lbl: idx for idx, lbl in map_b.items()},
                        ]
                        return get_reaction_template(fam, graphs,
                                                     labeled_per)
        return None

    return None


# ---------------------------------------------------------------------------
# Recipe application + product check (RMG _create_reaction product filter)
# ---------------------------------------------------------------------------

def _reactant_offsets(reactants):
    """The per-reactant atom offsets in the merged explicit graph (each
    reactant contributes its explicit-H atoms in AddHs order; the merged
    graph concatenates them)."""
    offsets = []
    off = 0
    for m in reactants:
        offsets.append(off)
        off += m._with_explicit_h().GetNumAtoms()
    return offsets


def _merged_labels(group, mapping, reactants, offsets):
    """Convert a merged-graph mapping {group_idx: merged_idx} (for a
    single-group family, e.g. R_Recombination) to a per-reactant label list:
    [{molecule_atom_index: recipe_label}, ...] (one dict per reactant).
    Returns None if a mapped merged index falls outside every reactant's
    range. The molecule's explicit-H atom order (AddHs) is the same order
    the merged graph uses, so a per-reactant index is merged_index - offset.
    Each group atom's OWN label is used (GroupAtom.label), so atoms sharing
    one label (Y_rad's two `*` atoms) are both labeled."""
    out = [{} for _ in reactants]
    for gi, mi in mapping.items():
        placed = False
        for ri, off in enumerate(offsets):
            n = reactants[ri]._with_explicit_h().GetNumAtoms()
            if off <= mi < off + n:
                lbl = group.atoms[gi].label
                if lbl:
                    out[ri][mi - off] = lbl
                placed = True
                break
        if not placed:
            return None
    return out
def _set_labels_on_mol(mol, mapping):
    """Set recipe labels {molecule_atom_index: label} on an rmgpu Molecule."""
    for idx, label in mapping.items():
        mol._rdkit.GetAtomWithIdx(int(idx)).SetProp('label', str(label))


def _apply_and_check(fam, reactants, lab, products):
    """Apply the family recipe to the labeled reactants and check the
    products reproduce `products`. `lab` is a list of
    {molecule_atom_index: label} dicts, one per reactant. Returns True iff the
    recipe yields products isomorphic (per piece) to `products` (RMG
    _create_reaction product check)."""
    fam02 = enum.Family(
        label=fam.label,
        recipe=fam.recipe,
        reverse_recipe=fam.reverse_recipe,
        own_reverse=fam.own_reverse,
        reversible=fam.reversible,
        allow_charged_species=fam.allow_charged_species,
        electrons=fam.electrons,
        reactant_num_effective=fam.reactant_num,
        product_num_forward=fam.product_num_forward,
        reverse_map=fam.reverse_map,
    )
    structures = []
    for i, m in enumerate(reactants):
        # Use the canonical RMG-adjlist form (RMG's explicit-H atom ordering)
        # so the recipe engine sees exactly the representation RMG used to
        # record the products (the step-02 reference reactants). The group
        # matcher's labelings are atom-indexed in the same ordering, so a
        # label set at index k lands on the atom RMG labeled, regardless of
        # whether the caller built the reactant from a SMILES (H-implicit) or
        # an RMG adjacency list.
        c = Molecule.from_adjacency_list(m.to_adjlist())
        if i < len(lab):
            _set_labels_on_mol(c, lab[i])
        structures.append(c)
    try:
        prod = apply_recipe(
            structures, fam.recipe,
            family_label=fam.label, forward=True,
            relabel_atoms=True, own_reverse=fam.own_reverse,
            reverse_map=fam.reverse_map,
            product_num=fam.product_num_forward,
            electrons=fam.electrons,
        )
    except (ActionError, KekulizationError):
        return False
    if not prod:
        return False
    try:
        return enum.same_species_lists(products, prod, strict=False)
    except Exception:
        return False
