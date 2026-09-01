"""
Product-enumeration orchestration (job-05/step-02).

Port of RMG-Py's reaction-generation orchestration (rmgpy/data/kinetics/
family.py: KineticsFamily.generate_reactions / _generate_reactions /
_generate_product_structures / _create_reaction / calculate_degeneracy,
and rmgpy/data/kinetics/common.py: find_degenerate_reactions /
reduce_same_reactant_degeneracy / check_for_same_reactants) onto rmgpu's
Molecule layer and the step-01 recipe engine (rmgpu/core/recipe.py
apply_recipe).

Structure
---------

- Family: a light holder for one family's ENUMERATION-relevant data (recipe,
  reverse recipe, own_reverse, reversible, allow_charged_species, electrons,
  the template-resolved reactant/product counts, reverse_map, the family
  label). Built by the family loader (step 04) or from the recorded
  reference (step-02 tests).
- TemplateReaction: the generated-reaction object (reactant/product
  Molecule pieces, degeneracy, direction, reversibility, template labels).
- generate_reactions(family, reactants, matcher, ...): the enumeration -
  resonance-expand the reactants, assign fresh atom IDs, iterate the
  matching atom labelings (supplied by `matcher`), apply the recipe
  (step-01 apply_recipe), build each reaction, and collapse the degenerate
  ones (find_degenerate_reactions) with the
  reduce_same_reactant_degeneracy adjustment.
- find_degenerate_reactions / reduce_same_reactant_degeneracy: exact port
  of the common.py degeneracy bookkeeping.
- calculate_degeneracy(family, reaction, resonance=True): port of
  KineticsFamily.calculate_degeneracy (re-enumerate with the products
  filter; requires family.matcher).
- RecordedMatcher: replays RMG-Py's own recorded atom labelings (the
  non-circular reference, gates/baselines/job05/step02_products_reference
  .json) - this step's matcher. The real group matcher lands in step 03
  (rmgpu/molecule/group.py + core/template.py) behind the same
  match_molecule(form, slot, branch) interface.

Scope (this step, gas-phase):
- unimolecular and bimolecular reactants; no surface sites, no
  termolecular, no charged-species families (their data formats arrive
  with the family loader, step 04);
- the FORWARD direction only. RMG also generates the reverse direction
  for reversible non-own-reverse families (self.reverse_template), but
  that requires the reverse family's template (labels + product count),
  which is step 04's loader's domain; none of this step's 30 recorded
  cases emit reverse-direction applications.

Degeneracy semantics (the subtle part - ported exactly)
--------------------------------------------------------
Each applied labeling produces a TemplateReaction (degeneracy 1). Two
generated reactions are combined (degeneracy summed) when their PRODUCT
pieces are isomorphic (strict=False, per piece, order-independent across
the pieces). An application contributes NO extra degeneracy when it is
IDENTICAL to one already in the group: same atom-ID sets per piece, and
the ID-pairing gives a valid element/bond/charge/radical mapping (RMG
Molecule.is_identical; atom IDs are tracked from the reactants through
merge -> apply_recipe -> split, exactly as RMG's atom.id). Reactions
with two (three) identical reactants have their degeneracy halved
(divided by 6) - the Bishop/Laidler correction
(reduce_same_reactant_degeneracy).

RMG's actual driver flow (rmgpy/data/kinetics/database.py
react_molecules + model loop) is:

    reactants, same_reactants = check_for_same_reactants(reactants)
    ensure_independent_atom_ids(reactants, resonance=True)   # resonance
    for combo in generate_molecule_combos(reactants):
        raw += family.generate_reactions(list(combo))
    collapsed = find_degenerate_reactions(raw, same_reactants, ...)

In rmgpu the resonance expansion and atom-ID assignment happen inside
generate_reactions (the equivalent of ensure_independent_atom_ids, which
RMG performs before the call) and the same-reactant check is on the
resonance-expanded species.
"""

from rdkit import Chem

from rmgpu.molecule.molecule import Molecule
from rmgpu.core.recipe import (
    ActionError,
    KekulizationError,
    ReactionRecipe,
    apply_recipe,
    clear_labeled_atoms,
)


# ---------------------------------------------------------------------------
# Family (enumeration data)
# ---------------------------------------------------------------------------

class Family:
    """
    One reaction family's enumeration-relevant data.

    Built by the family loader (step 04) from the family definition; the
    step-02 tests build it from the recorded reference. Mirrors the RMG
    KineticsFamily fields used by generate_reactions / apply_recipe:

    - label: the family label (lowercased, e.g. 'h_abstraction');
    - recipe / reverse_recipe: ReactionRecipe objects (the reverse recipe
      = the forward recipe undone);
    - own_reverse: the family is its own reverse (no separate reverse
      template) -> products are relabeled (step-01 relabel swap);
    - reversible: whether the family is reversible (stored on reactions);
    - allow_charged_species: charged products are kept when True, dropped
      when False;
    - electrons: free-electron count (0 for standard gas-phase families);
    - reactant_num_effective: the template's reactant count AFTER RMG's
      single-group split (e.g. R_Recombination's one Y_rad group -> 2);
    - product_num_forward: the effective product count of the forward
      template (RMG self.product_num or len(forward_template.products));
    - reverse_map: the family's reverseMap (relabel fallback);
    - matcher: the template matcher (step 03; RecordedMatcher in tests);
    - template_labels: the forward template's reactant labels (RMG
      get_reaction_template_labels; set by the loader).
    """

    def __init__(self, label, recipe, reverse_recipe=None, own_reverse=False,
                 reversible=False, allow_charged_species=False, electrons=0,
                 reactant_num_effective=1, product_num_forward=None,
                 reverse_map=None, matcher=None, template_labels=None,
                 forbidden=None):
        self.label = label.lower()
        self.recipe = recipe
        self.reverse_recipe = reverse_recipe
        self.own_reverse = own_reverse
        self.reversible = reversible
        self.allow_charged_species = allow_charged_species
        self.electrons = electrons
        self.reactant_num_effective = reactant_num_effective
        self.product_num_forward = product_num_forward
        self.reverse_map = reverse_map
        self.matcher = matcher
        self.template_labels = template_labels
        # RMG family's forbidden structures (list of {'label','group'} dicts
        # from the loader, or None). Checked against reactants and products
        # in _apply_one_application (RMG _generate_product_structures).
        self.forbidden = forbidden
        self._forbidden_groups = None  # lazily parsed + cached

    @classmethod
    def from_reference(cls, case):
        """
        Build a Family from a recorded reference case dict (the step-02
        reference JSON). The recorded fields are RMG's own values
        (family_meta), so this is a faithful construction.
        """
        meta = case['family_meta']
        recipe = ReactionRecipe.from_data(meta['recipe'])
        reverse = (ReactionRecipe.from_data(meta['reverse_recipe'])
                   if meta['reverse_recipe'] is not None else None)
        return cls(
            label=case['family'],
            recipe=recipe,
            reverse_recipe=reverse,
            own_reverse=meta['own_reverse'],
            reversible=meta['reversible'],
            allow_charged_species=meta['allow_charged_species'],
            electrons=meta['electrons'],
            reactant_num_effective=meta.get(
                'num_template_reactants_effective',
                meta['num_template_reactants']),
            product_num_forward=meta.get('effective_product_num_forward'),
            reverse_map=meta['reverse_map'],
        )


# ---------------------------------------------------------------------------
# TemplateReaction (a generated reaction)
# ---------------------------------------------------------------------------

class TemplateReaction(object):
    """
    A generated reaction (RMG TemplateReaction, the subset the
    enumeration/degeneracy machinery uses). reactants/products are lists of
    rmgpu Molecule pieces; the pieces keep their atom labels/IDs (the
    degeneracy machinery needs them; clear them with clear_labeled_atoms).
    """

    # The per-reaction template labels (list of str) or None (unset - the
    # family-level fallback is applied in generate_reactions). Set per
    # application in the fresh enumeration path (RMG
    # get_reaction_template_labels). Declared at class level so the
    # None->list reassignment type-checks.
    template = None  # type: list | None

    def __init__(self, reactants, products, degeneracy=1, reversible=False,
                 family=None, is_forward=True, electrons=0):
        self.reactants = reactants
        self.products = products
        self.degeneracy = degeneracy
        self.reversible = reversible
        self.family = family
        self.is_forward = is_forward
        self.electrons = electrons
        self.template = None
        self.duplicate = False

    def __repr__(self):
        return '<TemplateReaction %r -> %r deg=%s>' % (
            [m.get_formula() for m in self.reactants],
            [m.get_formula() for m in self.products], self.degeneracy)


# ---------------------------------------------------------------------------
# Piece comparisons (RMG Molecule.is_isomorphic / is_identical)
# ---------------------------------------------------------------------------

def piece_atom_ids(molecule):
    """The sorted tuple of atom IDs on a piece (the 'atomid' property)."""
    return tuple(sorted(int(a.GetProp('atomid'))
                        for a in molecule._rdkit.GetAtoms()
                        if a.HasProp('atomid')))


def piece_key(molecule):
    """
    Order-independent structural key of a piece: (element, radical, charge,
    sorted bond orders) per atom, sorted. Fast pre-filter for the
    isomorphic/identical comparisons (RMG's fingerprint quick check).
    """
    atoms = []
    for a in molecule._rdkit.GetAtoms():
        bonds = tuple(sorted(round(b.GetBondTypeAsDouble(), 6)
                             for b in a.GetBonds()))
        atoms.append((a.GetSymbol(), a.GetNumRadicalElectrons(),
                      a.GetFormalCharge(), bonds))
    return tuple(sorted(atoms))


def _with_explicit_h(mol):
    """
    Return a copy of `mol` with explicit hydrogens, without re-adding them
    when the molecule already carries explicit H atoms. rmgpu mols are the
    RMG representation (explicit H, kekulized) after apply_recipe /
    from_adjacency_list, but SMILES-built reactants are implicit-H; calling
    Chem.AddHs on an unsanitized mol throws (calcImplicitValence
    precondition), so sanitize the throwaway copy first.
    """
    m = Chem.Mol(mol)
    if any(atom.GetSymbol() == 'H' for atom in m.GetAtoms()):
        return m
    try:
        return Chem.AddHs(m)
    except Exception:
        try:
            Chem.SanitizeMol(m)
        except Exception:
            pass
        return Chem.AddHs(m)


def _element_multiset(rdkit_mol):
    """
    Element multiset of an explicit-H rdkit mol (the rmgpu representation),
    as a sorted tuple of (symbol, count). A robust isomorphism pre-filter that
    does not depend on RDKit's CalcMolFormula / sanitize (which can throw on a
    semi-kekulized aromatic piece).
    """
    from collections import Counter
    return tuple(sorted(Counter(a.GetSymbol()
                                for a in rdkit_mol.GetAtoms()).items()))


def _loose_isomorph(rdkit_mol_a, rdkit_mol_b):
    """
    Bond-order/electron-agnostic isomorphism (RMG is_isomorphic strict=False):
    same element multiset + same connectivity, ignoring bond order, radical
    electrons, charge and lone pairs. Normalizes both mols (explicit H, all
    bonds single, charge/radical zeroed) then does a bidirectional RDKit
    substructure match (which then reduces to element + connectivity).
    """
    def norm(mol):
        m = Chem.Mol(mol)
        for atom in m.GetAtoms():
            atom.SetFormalCharge(0)
            atom.SetNumRadicalElectrons(0)
        for bond in m.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
        return m
    ma = norm(_with_explicit_h(rdkit_mol_a))
    mb = norm(_with_explicit_h(rdkit_mol_b))
    if not ma.HasSubstructMatch(mb) or not mb.HasSubstructMatch(ma):
        return False
    return True


def _mols_isomorph(a, b, strict=True):
    """
    Isomorphism of two Molecule pieces (RMG Molecule.is_isomorphic, strict
    flag):
    - quick element-multiset pre-filter (explicit-H);
    - strict=True: RDKit bidirectional substructure match on the explicit-H
      forms (respects bond order) plus equal per-atom radical/charge
      multisets (RMG Atom.equivalent strict);
    - strict=False: element + connectivity only, ignoring bond order, radical
      electrons, charge and lone pairs (RMG is_isomorphic strict=False, which
      via VF2 uses Atom.equivalent strict=False = element only and only checks
      that edges exist).
    """
    ma = _with_explicit_h(a._rdkit)
    mb = _with_explicit_h(b._rdkit)
    if _element_multiset(ma) != _element_multiset(mb):
        return False
    if strict:
        ra = sorted((atom.GetNumRadicalElectrons(), atom.GetFormalCharge())
                    for atom in ma.GetAtoms())
        rb = sorted((atom.GetNumRadicalElectrons(), atom.GetFormalCharge())
                    for atom in mb.GetAtoms())
        if ra != rb:
            return False
        if not ma.HasSubstructMatch(mb) or not mb.HasSubstructMatch(ma):
            return False
        return True
    return _loose_isomorph(ma, mb)


def _mols_identical(a, b):
    """
    RMG Molecule.is_identical(strict=False): the two pieces carry the same
    set of atom IDs, and pairing atoms by (sorted) ID gives a valid mapping -
    same element per atom, same connectivity (every bonded pair in one is
    bonded in the other). Bond order, radical electrons and charge are
    IGNORED (strict=False): this is exactly the check find_degenerate_reactions
    uses for its `identical` flag (rxn0.is_isomorphic(rxn, check_identical=True,
    strict=False, check_template_rxn_products=True)). Ignoring bond order is
    what makes the two resonance forms produced by one recipe application
    identical, so a resonance-duplicate application contributes NO extra
    degeneracy (RMG's is_mapping_valid with strict=False skips the bond-order
    comparison and Atom.equivalent with strict=False compares element only).
    """
    if piece_atom_ids(a) != piece_atom_ids(b):
        return False
    am = {int(x.GetProp('atomid')): x
          for x in a._rdkit.GetAtoms() if x.HasProp('atomid')}
    bm = {int(x.GetProp('atomid')): x
          for x in b._rdkit.GetAtoms() if x.HasProp('atomid')}
    for aid, atom in am.items():
        if atom.GetSymbol() != bm[aid].GetSymbol():
            return False

    def connectivity(mol):
        edges = set()
        for bond in mol._rdkit.GetBonds():
            i1 = mol._rdkit.GetAtomWithIdx(bond.GetBeginAtomIdx())
            i2 = mol._rdkit.GetAtomWithIdx(bond.GetEndAtomIdx())
            if i1.HasProp('atomid') and i2.HasProp('atomid'):
                edges.add(frozenset((int(i1.GetProp('atomid')),
                                     int(i2.GetProp('atomid')))))
        return edges
    return connectivity(a) == connectivity(b)


def _permutations(n):
    import itertools
    return list(itertools.permutations(range(n)))


def same_species_lists(list1, list2, strict=True):
    """
    RMG same_species_lists: the two lists hold the same pieces in some
    order (permutations for 1-3 pieces). `strict` is the isomorphism flag.
    """
    if len(list1) != len(list2):
        return False
    n = len(list1)
    if n > 3:
        raise ValueError('Cannot compare lists of {0} species.'.format(n))
    for perm in _permutations(n):
        if all(_mols_isomorph(list1[i], list2[perm[i]], strict=strict)
               for i in range(n)):
            return True
    return False


def identical_species_lists(list1, list2):
    """
    The identical (atom-ID) counterpart of same_species_lists (RMG
    Reaction.is_isomorphic with check_identical=True): each piece of
    list1 pairs with an atom-ID-identical piece of list2 in some order.
    """
    if len(list1) != len(list2):
        return False
    n = len(list1)
    if n > 3:
        raise ValueError('Cannot compare lists of {0} species.'.format(n))
    for perm in _permutations(n):
        if all(_mols_identical(list1[i], list2[perm[i]])
               for i in range(n)):
            return True
    return False


# ---------------------------------------------------------------------------
# Same-reactant check + resonance (RMG check_for_same_reactants /
# ensure_independent_atom_ids / Species.is_isomorphic)
# ---------------------------------------------------------------------------

def _species_isomorph(species1, species2, strict=True):
    """RMG Species.is_isomorphic: any resonance form of any."""
    for m1 in species1:
        for m2 in species2:
            if _mols_isomorph(m1, m2, strict=strict):
                return True
    return False


def check_for_same_reactants(reactant_species):
    """
    RMG check_for_same_reactants: the number of mutually-isomorphic
    reactants (0, 1, 2 or 3). `reactant_species` is a list of species
    (each a list of Molecule resonance forms); the comparison is
    species-level (RMG Species.is_isomorphic).
    """
    if len(reactant_species) == 2:
        return 2 if _species_isomorph(reactant_species[0],
                                      reactant_species[1]) else 0
    if len(reactant_species) == 3:
        a, b, c = reactant_species
        s01 = _species_isomorph(a, b)
        s02 = _species_isomorph(a, c)
        if s01 and s02:
            return 3
        if s01 or s02:
            return 2
        if _species_isomorph(b, c):
            return 2
        return 0
    return 0


def expand_resonance(structures):
    """
    Expand each structure to its resonance set (RMG
    Species.generate_resonance_structures, keep_isomorphic=True). Returns a
    list of species (one per input), each a list of Molecule forms; the
    input form is the first (representative) form. rmgpu's resonance module
    (job-01) supplies the forms.
    """
    from rmgpu.molecule.resonance import generate_resonance_structures
    species = []
    for m in structures:
        try:
            forms = generate_resonance_structures(m)
        except Exception:
            forms = [m.copy(clear_labels=True)]
        if not forms:
            forms = [m.copy(clear_labels=True)]
        species.append(list(forms))
    return species


def assign_fresh_ids(species):
    """
    Assign atom IDs with RMG's resonance semantics and normalize every form to
    the explicit-H (RMG) representation. RMG's ensure_independent_atom_ids
    assigns fresh IDs to molecule[0] of each species (Molecule.assign_atom_ids),
    then regenerates the resonance structures, whose copies preserve atom.id -
    so every resonance form of a species SHARES the representative form's IDs
    positionally. Sharing is essential: it is what makes find_degenerate_
    reactions treat two products generated from two resonance forms of the same
    reactant as identical (no extra degeneracy). A fresh, unique ID per form
    would instead mark them non-identical and over-count the degeneracy.

    The explicit-H normalization is equally essential: RMG-Py stores molecules
    with explicit hydrogens (Molecule.from_smiles -> 10 atoms for C[CH]C), and
    the group matcher's match_molecule returns labelings in that explicit-H
    index space (via _explicit_graph / AddHs). rmgpu SMILES-built resonance
    forms are implicit-H (4 atoms for C[CH]C), so matching them directly
    yields out-of-range atom indices (Range Error) and the recipe engine sees
    the wrong representation. Rebuilding each form through its adjacency list
    (to_adjlist / from_adjacency_list) puts it in RMG's explicit-H, kekulized
    representation - the space the matcher, the recipe engine and the degeneracy
    bookkeeping all operate in.
    """
    base = 0
    for i, spc in enumerate(species):
        # Normalize to explicit-H first so the ID count and the matcher/recipe
        # index spaces agree.
        species[i] = [Molecule.from_adjacency_list(f.to_adjlist()) for f in spc]
        n = len(species[i][0]._rdkit.GetAtoms())
        ids = list(range(base, base + n))
        for form in species[i]:
            for j, atom in enumerate(form._rdkit.GetAtoms()):
                if j < len(ids):
                    atom.SetProp('atomid', str(ids[j]))
        base += n


# ---------------------------------------------------------------------------
# Reaction construction (RMG _create_reaction)
# ---------------------------------------------------------------------------

def _is_balanced(reactants, products):
    """
    RMG Reaction.is_balanced: same element counts and same net charge on
    both sides. Counts are taken on the explicit-H (RMG) representation.
    """
    from collections import Counter

    def counts(mols):
        c = Counter()
        charge = 0
        for m in mols:
            explicit = _with_explicit_h(m._rdkit)
            for a in explicit.GetAtoms():
                c[a.GetSymbol()] += 1
            charge += m.get_charge()
        return c, charge

    ca, qa = counts(reactants)
    cb, qb = counts(products)
    return ca == cb and qa == qb


def create_reaction(family, reactant_structures, product_structures, forward):
    """
    RMG KineticsFamily._create_reaction: build a TemplateReaction or None
    (products identical to reactants, a charged species in a family that
    does not allow charged species, or an unbalanced reaction).
    """
    if same_species_lists(reactant_structures, product_structures,
                          strict=True):
        return None
    if forward:
        reactants, products = reactant_structures, product_structures
    else:
        reactants, products = product_structures, reactant_structures
    if not family.allow_charged_species:
        for m in reactants + products:
            if m.get_charge() != 0:
                return None
    if not _is_balanced(reactants, products):
        return None
    return TemplateReaction(
        reactants=reactants,
        products=products,
        degeneracy=1,
        reversible=family.reversible,
        family=family.label,
        is_forward=forward,
        electrons=family.electrons,
    )


# ---------------------------------------------------------------------------
# Degeneracy bookkeeping (RMG common.py)
# ---------------------------------------------------------------------------

def find_degenerate_reactions(rxn_list, same_reactants=None,
                              template=None, family=None, resonance=True):
    """
    RMG common.find_degenerate_reactions, ported line-for-line (the
    gas-phase enumeration path; the save_order/kinetics_database args of
    RMG are dropped - save_order is False here and the family instance is
    passed directly).

    1. Optional template filter: keep only reactions whose frozenset of
       template labels equals the given frozenset (RMG also returns [] with
       a log when the filter matches nothing; rmgpu returns [] the same way
       - no log).
    2. Sort the reactions into sublists: for each rxn0, scan the existing
       sublists in order; a sublist ISOMORPHS rxn0 when ANY of its members
       has isomorphic products (strict=False, RMG check_template_rxn_products
       -> same_species_lists on the product pieces). Within an isomorphic
       sublist: if ANY member is IDENTICAL (atom-ID, strict=False) to rxn0,
       rxn0 contributes no extra degeneracy and the scan stops; else if the
       FIRST isomorphic member has the same template, rxn0 joins that
       sublist; else (different template) rxn0 and the sublist head are
       marked duplicate and the scan CONTINUES to the remaining sublists.
       No isomorphic sublist -> rxn0 starts a new sublist (for/else).
    3. Collapse: each sublist yields its head with the summed degeneracy.
    4. reduce_same_reactant_degeneracy on the forward reactions; reverse
       reactions of non-own-reverse families would be recomputed via
       calculate_degeneracy (ported, though this step's forward-only
       enumeration does not emit such reactions).
    """
    selected = rxn_list
    if template is not None:
        selected = []
        template = frozenset(template)
        for rxn in rxn_list:
            if frozenset(rxn.template) == template:
                selected.append(rxn)
        if not selected:
            return []

    groups = []
    for rxn0 in selected:
        identical = False
        for sub in groups:
            isomorphic = False
            same_template = True
            for rxn in sub:
                isomorphic = same_species_lists(
                    rxn0.products, rxn.products, strict=False)
                if isomorphic:
                    identical = identical_species_lists(
                        rxn0.products, rxn.products)
                    if identical:
                        # an exact copy is already in this sublist
                        break
                    same_template = (frozenset(rxn.template) ==
                                     frozenset(rxn0.template))
                else:
                    # this sublist has a different product
                    break
            if identical:
                # no extra degeneracy
                break
            elif isomorphic:
                if same_template:
                    # the right sublist, no identical reaction: join it
                    sub.append(rxn0)
                    break
                else:
                    # isomorphic but different template: keep as a separate
                    # duplicate reaction, keep searching the other sublists
                    rxn0.duplicate = True
                    sub[0].duplicate = True
                    continue
        else:
            groups.append([rxn0])

    collapsed = []
    for sub in groups:
        rxn = sub[0]
        rxn.degeneracy = sum(r.degeneracy for r in sub)
        collapsed.append(rxn)

    for rxn in collapsed:
        if rxn.is_forward:
            reduce_same_reactant_degeneracy(rxn, same_reactants)
        elif family is not None and not family.own_reverse:
            rxn.degeneracy = calculate_degeneracy(family, rxn,
                                                  resonance=resonance)
    return collapsed


def reduce_same_reactant_degeneracy(reaction, same_reactants=None):
    """
    RMG common.reduce_same_reactant_degeneracy (Bishop & Laidler 1965),
    ported exactly: reactions with identical reactants overcount the
    translational transition states - halve (2 reactants) or divide by 6
    (3 reactants) the degeneracy. The 2-reactant test runs for BOTH
    directions (isomorphic reactant molecules); the 3-reactant reduction
    applies to forward reactions via the precomputed count, or (reverse)
    via direct isomorphism checks.
    """
    if not (same_reactants == 0 or same_reactants == 1):
        if len(reaction.reactants) == 2:
            if ((reaction.is_forward and same_reactants == 2) or
                    _mols_isomorph(reaction.reactants[0],
                                   reaction.reactants[1])):
                reaction.degeneracy *= 0.5
        elif len(reaction.reactants) == 3:
            if reaction.is_forward:
                if same_reactants == 3:
                    reaction.degeneracy /= 6.0
                elif same_reactants == 2:
                    reaction.degeneracy *= 0.5
            else:
                a, b, c = reaction.reactants
                s01 = _mols_isomorph(a, b)
                s02 = _mols_isomorph(a, c)
                if s01 and s02:
                    reaction.degeneracy /= 6.0
                elif s01 or s02:
                    reaction.degeneracy *= 0.5
                elif _mols_isomorph(b, c):
                    reaction.degeneracy *= 0.5


# ---------------------------------------------------------------------------
# The enumeration
# ---------------------------------------------------------------------------

def _reaction_template(tfam, reactant_mols, per_mol_mappings):
    """
    Compute the per-application template labels (RMG
    KineticsGroups.get_reaction_template: descend each deduped forward-
    template top node to its most-specific node, given the labeled
    reactants). This is essential for find_degenerate_reactions: two
    reactions with the same product but DIFFERENT templates are kept as
    separate duplicate reactions (degeneracy NOT summed), so the per-
    reaction template is what separates e.g. intra_H_migration's two
    [CH2]CCCC products (Cs_H_out_2H reached via R2H_S vs via R4H_SSS).

    `reactant_mols` is the list of labeled reactant Molecules (RMG
    reactant order); `per_mol_mappings` is a parallel list of
    {molecule_atom_index: recipe_label} dicts (the labeling applied to each).
    Returns the list of template labels, or None when the tree cannot descend
    (RMG get_reaction_template -> None; the caller then falls back to the
    family-level template).
    """
    from rmgpu.core.template import (
        _dedup_top, get_reaction_template, merge_graph, _reactant_offsets,
    )
    from rmgpu.molecule.group import Group, _explicit_graph
    deduped = _dedup_top(tfam)
    if len(deduped) == 1 and len(reactant_mols) > 1 and \
            isinstance(deduped[0].item, Group):
        # single-group split: one deduped top, merged graph
        graph = merge_graph(reactant_mols)
        offsets = _reactant_offsets(reactant_mols)
        merged_lab = {}
        for i, mol in enumerate(reactant_mols):
            for idx, lbl in per_mol_mappings[i].items():
                merged_lab[lbl] = idx + offsets[i]
        try:
            return get_reaction_template(tfam, [graph], [merged_lab])
        except Exception:
            return None
    # unimolecular / bimolecular: one graph per reactant
    graphs = [_explicit_graph(m) for m in reactant_mols]
    labeled_per = [{lbl: idx for idx, lbl in mp.items()}
                   for mp in per_mol_mappings]
    try:
        return get_reaction_template(tfam, graphs, labeled_per)
    except Exception:
        return None


def _apply_one_application(family, structures, relabel_atoms):
    """
    Apply the recipe to ONE atom-labeling application (a list of labeled,
    atom-ID-tagged reactant Molecules, in RMG reactant order) through the
    step-01 engine. Returns the created TemplateReaction, or None when the
    application yields no reaction (RMG _generate_product_structures ->
    None on InvalidActionError/KekulizationError/empty products, or
    _create_reaction -> None on identical/charged/unbalanced products).
    `structures` are used as-is (the engine copies them; the originals are
    untouched), so the caller may reuse them.

    RMG _generate_product_structures also rejects the application when any
    REACTANT (with its template labels set) or any PRODUCT piece matches a
    family forbidden structure (is_molecule_forbidden, a label-aligned
    subgraph check). That check is what drops e.g. the diradical-forming
    H_abstraction applications (the disprop1_base_case forbidden group
    matches a closed-shell carbon labeled *1 bonded to an H labeled *2 AND
    a radical atom; only the terminal C-H labelings of an alkyl radical
    carry that pattern, so only the central radical-site application
    survives). It is run on the labeled reactants BEFORE the recipe (RMG
    order) and on each product piece AFTER it.
    """
    forbidden_groups = _family_forbidden_groups(family)
    if forbidden_groups:
        # RMG: check the labeled reactant structures (the labels are already
        # set by the caller before this is invoked).
        for s in structures:
            if is_molecule_forbidden(s, forbidden_groups):
                return None
    try:
        product_structures = apply_recipe(
            structures,
            family.recipe,
            family_label=family.label,
            forward=True,
            relabel_atoms=relabel_atoms,
            own_reverse=family.own_reverse,
            reverse_map=family.reverse_map,
            product_num=family.product_num_forward,
            electrons=family.electrons,
        )
    except (ActionError, KekulizationError):
        return None  # RMG: InvalidActionError / KekulizationError -> no products
    if not product_structures:
        return None  # product-count / net-charge mismatch -> not a match
    if forbidden_groups:
        # RMG: reject when any product piece matches a forbidden structure.
        for p in product_structures:
            if is_molecule_forbidden(p, forbidden_groups):
                return None
    return create_reaction(family, structures, product_structures, True)


# ---------------------------------------------------------------------------
# Forbidden-structure check (RMG ForbiddenStructures.is_molecule_forbidden)
# ---------------------------------------------------------------------------

def _family_forbidden_groups(family):
    """Lazily parse + cache the family's forbidden Groups (the loader stores
    them as {'label', 'group'} text dicts). Returns [] when the family has
    none (or the cached result on subsequent calls)."""
    cached = getattr(family, '_forbidden_groups', None)
    if cached is not None:
        return cached
    if not getattr(family, 'forbidden', None):
        groups = []
    else:
        from rmgpu.molecule.group import Group
        groups = []
        for fb in family.forbidden:
            text = fb['group'] if isinstance(fb, dict) else fb
            text = text.strip()
            if text.startswith(('OR{', 'AND{', 'NOT OR{', 'NOT AND{')):
                # LogicNode forbidden group (none in the default set); the
                # subgraph check is Groups-only. Skip (documented gap).
                continue
            groups.append(Group().parse(text))
    try:
        family._forbidden_groups = groups
    except Exception:
        pass  # read-only family; cache is best-effort
    return groups


def _clean_explicit(mol):
    """A sanitized, explicit-H rmgpu Molecule equal to `mol`, for the
    group matcher (which needs a kekulized explicit-H graph with computed
    implicit valence). `mol` may already be explicit-H (post-recipe product)
    or implicit-H (a labeled reactant form); AddHs on an unsanitized
    semi-kekulized mol throws, so sanitize a throwaway copy first. Labels
    are preserved (Chem.Mol copy keeps atom properties)."""
    m = Chem.Mol(mol._rdkit)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    if not any(a.GetSymbol() == 'H' for a in m.GetAtoms()):
        try:
            m = Chem.AddHs(m)
        except Exception:
            try:
                Chem.SanitizeMol(m)
            except Exception:
                pass
            m = Chem.AddHs(m)
    return Molecule._from_rdmol(m)


def is_molecule_forbidden(molecule, forbidden_groups):
    """
    RMG ForbiddenStructures.is_molecule_forbidden: True if `molecule`
    matches any forbidden `Group`. The check is LABEL-ALIGNED (RMG
    "labeled atoms on the forbidden structures and the molecule are
    honored"): every label on the group must be present on the molecule,
    and the subgraph search is seeded with the same-labeled atoms paired
    (RMG Molecule.is_subgraph_isomorphic(generate_initial_map=True)). A
    molecule atom may match a group atom only via a same-label pairing;
    unlabeled group atoms are free to match any feasible atom. Returns True
    on the first matching group.

    `forbidden_groups` is a list of parsed Group objects (the family's
    forbidden structures).
    """
    import itertools
    from rmgpu.molecule.group import match_explicit_graph, _explicit_graph
    mlabeled = molecule.get_all_labeled_atoms()
    graph = _explicit_graph(_clean_explicit(molecule))
    for group in forbidden_groups:
        glabeled = group.get_all_labeled_atoms()  # {label: group_atom_idx}
        # Every group label must be present in the molecule.
        if any(label not in mlabeled for label in glabeled):
            continue
        # Seed the subgraph search: unique molecule atoms per label are fixed
        # pairings; non-unique labels are cross-products (RMG's atms loop).
        fixed = {}
        keys = []
        choices = []
        for label, group_idx in glabeled.items():
            v = mlabeled[label]
            if isinstance(v, list):
                keys.append(group_idx)
                choices.append(v)
            else:
                fixed[group_idx] = v
        found = False
        if choices:
            for combo in itertools.product(*choices):
                if len(set(combo)) != len(combo):
                    continue  # two group atoms to the same molecule atom
                initial_map = dict(fixed)
                for group_idx, mol_idx in zip(keys, combo):
                    initial_map[group_idx] = mol_idx
                if match_explicit_graph(graph, group, initial_map=initial_map):
                    found = True
                    break
        else:
            if match_explicit_graph(graph, group, initial_map=fixed):
                found = True
        if found:
            return True
    return False


def _set_labels(structure, mapping):
    """Set template labels {atom_index: label} on `structure` (a Molecule)."""
    for atom_index, label in mapping.items():
        structure._rdkit.GetAtomWithIdx(int(atom_index)).SetProp(
            'label', str(label))


def _match_labelings(group, form):
    """Match a concrete Group to a resonance `form` and return the list of
    labelings ({molecule_atom_index: recipe_label}), one per valid subgraph
    isomorphism. Mirrors RMG _match_reactant_to_template + the labeling
    inversion used by the single-group-split branch (where the template
    reactant is a split component, not a forward-template slot, so the
    matcher's match_molecule slot interface does not apply)."""
    from rmgpu.molecule.group import match_explicit_graph, _explicit_graph
    graph = _explicit_graph(form)
    out = []
    for m in match_explicit_graph(graph, group):
        lab = {}
        for gi, mi in m.items():
            lbl = group.atoms[gi].label
            if lbl:
                lab[mi] = lbl
        out.append(lab)
    return out


def _enumerate_fresh(family, reactants, matcher, relabel_atoms):
    """
    The RMG enumeration loop over resonance forms x template matchings x
    the branches (unimolecular; bimolecular A+B / B+A; single-group split),
    driven by a matcher that supplies FRESH labelings via
    match_molecule(form, slot, branch) (step 03's group matcher). This is the
    forward-direction subset of RMG _generate_reactions.

    expand_resonance + assign_fresh_ids normalize every resonance form to the
    explicit-H (RMG) representation and assign RMG's shared-per-species atom
    IDs before matching, so the matcher's labelings, the recipe application
    and the degeneracy bookkeeping all operate in the same index space.
    """
    from rmgpu.molecule.group import Group
    species = expand_resonance(reactants)
    assign_fresh_ids(species)
    tfam = matcher.family
    rxn_list = []

    # ---- single-group split (Disproportionation / R_Recombination) ----
    # One forward-template reactant that splits into >1 component groups; each
    # species matches one component (RMG _generate_reactions split branch).
    top = tfam.top
    if (len(top) == 1 and family.reactant_num_effective > 1
            and isinstance(top[0].item, Group)):
        comps = top[0].item.split()
        if len(species) == len(comps) == 2:
            (sa, sb), (ca, cb) = species, comps
            # A + B branch (template order: A->ca, B->cb; RMG passes
            # structures [molecule_b, molecule_a], maps [map_b, map_a])
            for ma in sa:
                for mb in sb:
                    la = _match_labelings(ca, ma)
                    lb = _match_labelings(cb, mb)
                    for map_a in la:
                        for map_b in lb:
                            _set_labels(ma, map_a)
                            _set_labels(mb, map_b)
                            rxn = _apply_one_application(
                                family, [mb, ma], relabel_atoms)
                            if rxn is not None:
                                rxn.template = _reaction_template(
                                    tfam, [ma, mb], [map_a, map_b])
                                rxn_list.append(rxn)
                            clear_labeled_atoms([ma, mb])
            # B + A (swapped) - only when the two reactants differ (the swap
            # is redundant when the species are isomorphic)
            if not _species_isomorph(sa, sb):
                for ma in sa:
                    for mb in sb:
                        la = _match_labelings(cb, ma)
                        lb = _match_labelings(ca, mb)
                        for map_a in la:
                            for map_b in lb:
                                _set_labels(ma, map_a)
                                _set_labels(mb, map_b)
                                rxn = _apply_one_application(
                                    family, [ma, mb], relabel_atoms)
                                if rxn is not None:
                                    rxn.template = _reaction_template(
                                        tfam, [ma, mb], [map_a, map_b])
                                    rxn_list.append(rxn)
                                clear_labeled_atoms([ma, mb])
        return rxn_list

    # ---- unimolecular (one reactant, single template slot) ----
    if len(species) == 1:
        for form in species[0]:
            for mapping in matcher.match_molecule(form, 0, 'ab'):
                _set_labels(form, mapping)
                rxn = _apply_one_application(family, [form], relabel_atoms)
                if rxn is not None:
                    rxn.template = _reaction_template(
                        tfam, [form], [mapping])
                    rxn_list.append(rxn)
                clear_labeled_atoms([form])
    else:
        a, b = species
        # A + B (template order; RMG passes structures [molecule_b,
        # molecule_a], maps [map_b, map_a])
        for ma in a:
            for mb in b:
                for map_a in matcher.match_molecule(ma, 0, 'ab'):
                    for map_b in matcher.match_molecule(mb, 1, 'ab'):
                        _set_labels(ma, map_a)
                        _set_labels(mb, map_b)
                        rxn = _apply_one_application(
                            family, [mb, ma], relabel_atoms)
                        if rxn is not None:
                            rxn.template = _reaction_template(
                                tfam, [mb, ma], [map_b, map_a])
                            rxn_list.append(rxn)
                        clear_labeled_atoms([ma, mb])
        # B + A (swapped) - only when the two reactants differ
        if not _species_isomorph(a, b):
            for ma in a:
                for mb in b:
                    for map_a in matcher.match_molecule(ma, 1, 'ba'):
                        for map_b in matcher.match_molecule(mb, 0, 'ba'):
                            _set_labels(ma, map_a)
                            _set_labels(mb, map_b)
                            rxn = _apply_one_application(
                                family, [ma, mb], relabel_atoms)
                            if rxn is not None:
                                rxn.template = _reaction_template(
                                    tfam, [ma, mb], [map_a, map_b])
                                rxn_list.append(rxn)
                            clear_labeled_atoms([ma, mb])
    return rxn_list


def generate_reactions(family, reactants, matcher=None, products=None,
                       prod_resonance=True, delete_labels=True,
                       relabel_atoms=True, template=None):
    """
    Generate all FORWARD-direction reactions of `family` between the given
    `reactants` (RMG KineticsFamily.generate_reactions, gas-phase
    unimolecular / bimolecular subset; the reverse direction is step 04's
    scope - see the module docstring). Returns the collapsed list of
    TemplateReaction objects (degenerate applications combined, same-reactant
    reduction applied).

    Args:
        family: a Family (the enumeration data).
        reactants: a list of 1 or 2 rmgpu Molecule reactants (one species
            each).
        matcher: an object with EITHER
            - `yield_applications()` -> iterator of applications, each a list
              of labeled, atom-ID-tagged reactant Molecules ready for
              apply_recipe (the RecordedMatcher; RMG's own recorded
              applications are replayed), OR
            - `match_molecule(form, slot, branch)` -> list of labelings (dicts
              {atom_index: label}) of `form` against template reactant `slot`
              for reactant order `branch` ('ab' = A+B / template order, 'ba' =
              the swapped B+A order) - the step-03 group matcher.
            None -> falls back to `family.matcher` (the RecordedMatcher in
            this step; the group matcher once step 03 lands). If that is
            also None, no reactions.
        products: optional list of Molecule desired products; when given, only
            reactions that produce them are kept (strict per piece, or
            resonance-relaxed per `prod_resonance`) - RMG's products filter.
        prod_resonance: RMG flag (the filter uses strict=not prod_resonance).
        delete_labels: when True, clear the atom labels on the returned
            reactions' pieces (RMG delete_labels; the degeneracy machinery
            runs before the clearing).
        relabel_atoms: whether self-reverse families relabel their products
            (RMG flag; True default).
        template: optional list of template labels; when given, only
            reactions whose template equals it (as a frozenset) are kept
            after the collapse (RMG find_degenerate_reactions template
            filter; used by calculate_degeneracy).
    """
    if matcher is None:
        matcher = family.matcher
    if matcher is None:
        return []
    reactants = list(reactants)
    if len(reactants) not in (1, 2):
        raise ValueError('generate_reactions supports 1 or 2 reactants '
                         '(got {0}).'.format(len(reactants)))

    # Same-reactant count (species-level; RMG check_for_same_reactants on the
    # resonance-expanded species).
    same_reactants = check_for_same_reactants(expand_resonance(reactants))

    if hasattr(matcher, 'yield_applications'):
        # Replay path (RecordedMatcher): RMG's own recorded applications in
        # RMG's own order. Every application is replayed; the ones that
        # produce a reaction are, in order, exactly RMG's raw reactions, so
        # the recorded raw_templates (RMG raw order) zip onto them by index.
        # Attaching the per-raw template here is essential: find_degenerate_
        # reactions keeps isomorphic-but-different-template products as
        # separate duplicate reactions instead of combining their degeneracy.
        raw_templates = getattr(matcher, 'case', {}).get('raw_templates')
        rxn_list = []
        tindex = 0
        for structures in matcher.yield_applications():
            if len(structures) != family.reactant_num_effective:
                continue
            # Copy the structures: apply_recipe / the degeneracy machinery
            # mutate reactant pieces in place (labels cleared with
            # delete_labels=True), and the matcher's structures must stay
            # reusable across re-runs (e.g. calculate_degeneracy).
            rxn = _apply_one_application(
                family, [m.copy(clear_labels=False) for m in structures],
                relabel_atoms)
            if rxn is not None:
                if (raw_templates is not None
                        and tindex < len(raw_templates)):
                    rxn.template = list(raw_templates[tindex])
                tindex += 1
                rxn_list.append(rxn)
    else:
        if len(reactants) != family.reactant_num_effective:
            return []
        rxn_list = _enumerate_fresh(family, reactants, matcher, relabel_atoms)

    # RMG's products filter (only when a product set is requested)
    if products is not None:
        kept = []
        for rxn in rxn_list:
            prods0 = rxn.products if rxn.is_forward else rxn.reactants
            if same_species_lists(products, prods0,
                                  strict=not prod_resonance):
                kept.append(rxn)
        rxn_list = kept

    # Template labels (RMG get_reaction_template_labels). The replay path has
    # already set each raw reaction's template from the reference; the fresh
    # (step-03 group matcher) path sets them per application. Only fill in a
    # family-level fallback where the template is still unset.
    if family.template_labels is not None:
        for rxn in rxn_list:
            if rxn.template is None:
                rxn.template = list(family.template_labels)

    # Degeneracy (RMG: one find_degenerate_reactions over the whole raw list).
    collapsed = find_degenerate_reactions(
        rxn_list, same_reactants=same_reactants, template=template,
        family=family, resonance=True)

    if delete_labels:
        for rxn in collapsed:
            for m in list(rxn.reactants) + list(rxn.products):
                clear_labeled_atoms([m])
    return collapsed


# ---------------------------------------------------------------------------
# calculate_degeneracy (RMG KineticsFamily.calculate_degeneracy)
# ---------------------------------------------------------------------------

class DegeneracyError(Exception):
    """RMG KineticsError raised by calculate_degeneracy on mismatch."""


def calculate_degeneracy(family, reaction, resonance=True):
    """
    RMG KineticsFamily.calculate_degeneracy, ported exactly: for a `reaction`
    (a TemplateReaction with Molecule reactants/products) in the direction in
    which the kinetics are defined, re-enumerate it - every application of
    the family to the reaction's reactants (all resonance forms and the
    bimolecular swap branches), filtered to the reaction's own products -
    then collapse with the TEMPLATE filter on the reaction's own template
    and the same-reactant count of the reaction's reactants, and return the
    collapsed degeneracy. A mismatch (not exactly one collapsed reaction)
    raises DegeneracyError, as RMG raises KineticsError. Requires
    family.matcher.
    """
    if family.matcher is None:
        raise ValueError(
            'calculate_degeneracy requires family.matcher to be set.')
    reactants = [m.copy(clear_labels=True) for m in reaction.reactants]
    rxns = generate_reactions(
        family, reactants, matcher=family.matcher,
        products=reaction.products, prod_resonance=resonance,
        delete_labels=False, template=reaction.template)
    if len(rxns) != 1:
        raise DegeneracyError(
            'Unable to calculate degeneracy for reaction {0!r}: expected '
            '1 reaction, generated {1}.'.format(reaction, len(rxns)))
    return rxns[0].degeneracy


# ---------------------------------------------------------------------------
# RecordedMatcher (this step's matcher - replays RMG-Py's labelings)
# ---------------------------------------------------------------------------

class RecordedMatcher:
    """
    A matcher that replays RMG-Py's own recorded applications (the step-02
    reference, gates/baselines/job05/step02_products_reference.json).

    Design
    ------
    The non-circular reference for this step is RMG-Py's own enumeration:
    for each case the reference records every `_generate_product_structures`
    call as the LABELED reactant structures exactly as RMG applied them (labels
    on atoms, in RMG's reactant order, with RMG's exact per-atom IDs). Replaying
    those applications through rmgpu's own `apply_recipe` is the faithful test
    of the port: same labeled reactant graph, same recipe -> same product
    structure, and the same reactant atom-ID sets flow into the products, so
    the identical/isomorphic degeneracy distinction is reproduced exactly.

    Interface (the hook step 03's real group matcher will implement):
      - `yield_applications()` -> iterator of applications, each a list of
        ready-to-apply reactant structures (Molecules with their template
        labels and atom IDs set). This is the primary path; `generate_reactions`
        iterates these and applies the recipe.
      - `match_molecule(form, slot, branch)` -> list of labelings (kept as the
        step-03 interface for a matcher that produces FRESH labelings over
        resonance forms; the recorded matcher returns [] since it replays).

    The recorded application order, reactant order within each application, and
    per-atom IDs are RMG's own, so no reconstruction is needed: each application
    is applied verbatim.
    """

    def __init__(self, case):
        """
        Args:
            case: the recorded case dict (family, reactant_smiles,
                applications).
        """
        self.case = case
        # Keep the RAW adjacency texts + atom IDs, not built Molecules:
        # apply_recipe / clear_labeled_atoms mutate the structures in place
        # (labels, bond orders), and the enumeration may be re-run on the
        # same matcher (e.g. calculate_degeneracy after generate_reactions),
        # so each replay must start from pristine structures.
        self._apps = []
        for app in case['applications']:
            if not app.get('ok', True):
                continue
            self._apps.append((app['labeled_reactants'],
                               app['reactant_atom_ids']))

    def _structures(self, texts, idlists):
        structures = [Molecule.from_adjacency_list(text)
                      for text in texts]
        # Set RMG's exact per-atom IDs (atom order matches the adjlist).
        for struct, idlist in zip(structures, idlists):
            for atom, aid in zip(struct._rdkit.GetAtoms(), idlist):
                atom.SetProp('atomid', str(int(aid)))
        return structures

    def yield_applications(self):
        """Iterate the recorded applications (lists of labeled, ID-tagged
        reactant structures, ready for apply_recipe). Each call yields
        FRESH structures (apply_recipe mutates them)."""
        return iter(self._structures(texts, idlists)
                    for texts, idlists in self._apps)

    def match_molecule(self, form, slot, branch='ab'):
        """Step-03 interface (fresh labelings over resonance forms). The
        recorded matcher replays pre-built applications instead, so this
        returns []. Step 03's group matcher will implement this."""
        return []
