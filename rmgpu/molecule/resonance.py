"""Resonance structure generation for rmgpu, implementing RMG-style rules."""

from rdkit import Chem
from rdkit.Chem import BondType
from rmgpu.molecule.molecule import Molecule


def _bond_order_str(order):
    """Map bond order numeric value to a string for pattern matching."""
    if order == 1.5:
        return 'A'
    return {1: 'S', 2: 'D', 3: 'T'}.get(order, 'S')


def _is_radical(mol):
    """Return True if molecule has radical electrons."""
    return any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms())


# ---------------------------------------------------------------------------
# Structural (positional) keys + aromatic-representative detection
# ---------------------------------------------------------------------------

def _form_key(m):
    """A STRUCTURAL dedup key for a resonance form: the exact bond-order
    signature on the STORED mol, (atom_a, atom_b, order) tuples, plus the
    per-atom radical/charge.

    Bond orders are NORMALIZED so that an aromatic-flagged bond (RDKit
    `GetIsAromatic()`) counts as 1.5 regardless of whether RDKit shows it as a
    fractional 1.5 or as an alternating single/double with the aromatic flag
    set. rmgpu's aromatic species exist in two equivalent representations:
    the delocalized one (1.5 bonds, aromatic-flagged, from SetAromaticity) and
    the flagged Kekulé one (S/D bonds + aromatic flag, as parsed from
    `c1ccccc1`). Both are the SAME aromatic representative and must dedupe to
    one form, while the genuinely-kekulized variant (S/D, NO aromatic flag)
    stays distinct. This is what keeps benzene/toluene at 2 forms (aromatic +
    kekulized, matching the job-01 reference) AND the 5-form benzylic set
    (job-05 Intra_ene).

    The key is NOT the canonical SMILES (the two ortho forms share one SMILES)
    nor the adjlist string (RDKit serializes a 1.5 aromatic ring bond as '2',
    collapsing aromatic + kekulized).

    CRITICAL: the signature is computed on the STORED mol (implicit-H,
    `m._rdkit`), NOT `_with_explicit_h()`: RDKit's AddHs/sanitize RE-AROMATIZE
    a genuinely-kekulized benzene ring (re-setting the aromatic flag), which
    would collapse the kekulized variant (S/D, no flag) into the aromatic
    representative (S/D, flagged) and drop it. The stored mol preserves the
    flag distinction.
    """
    try:
        em = m._rdkit
    except Exception:
        return m.to_smiles()
    bonds = []
    for b in em.GetBonds():
        o = b.GetBondTypeAsDouble()
        if b.GetIsAromatic():
            o = 1.5
        if o in (1.0, 1.5, 2.0, 3.0):
            bonds.append((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                          max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), o))
    atoms = []
    for a in em.GetAtoms():
        atoms.append((a.GetIdx(), a.GetSymbol(), a.GetNumRadicalElectrons(),
                      a.GetFormalCharge()))
    return (tuple(sorted(bonds)), tuple(atoms))


def _aromatic_rep(mol):
    """Return (is_aromatic_rep, ring_bond_pairs) for `mol`'s explicit-H graph.

    `is_aromatic_rep` is True iff `mol` has a 6-membered ring ALL of whose
    bonds carry RDKit's aromatic flag - the delocalized aromatic
    representative. RDKit shows this form either as fractional 1.5 bonds
    (after SetAromaticity) or as alternating S/D bonds WITH the aromatic flag
    (as parsed from `c1ccccc1`); both are the same representative, so the test
    is the aromatic FLAG, not the bond order. The genuinely-kekulized variant
    (S/D, no aromatic flag) is NOT an aromatic representative.

    `ring_bond_pairs` is the frozenset of (min_idx, max_idx) ring bonds of the
    first 6-membered ring (empty if none), captured BEFORE any adjlist
    round-trip so the aromatic-representative flag can be re-applied after the
    round-trip Kekulizes the ring (job-05 step-06, Fix 3).

    Computed on the STORED mol (`mol._rdkit`), not the AddHs graph: AddHs /
    sanitize RE-AROMATIZE a genuinely-kekulized benzene ring (re-setting the
    aromatic flag), which would make the kekulized variant look like the
    aromatic representative. The stored mol preserves the flag.
    """
    try:
        em = mol._rdkit
    except Exception:
        return False, frozenset()
    for ring in em.GetRingInfo().AtomRings():
        if len(ring) != 6:
            continue
        pairs = set()
        all_ar = True
        for k in range(6):
            a, b = ring[k], ring[(k + 1) % 6]
            bd = em.GetBondBetweenAtoms(a, b)
            if bd is None or not bd.GetIsAromatic():
                all_ar = False
                break
            pairs.add((min(a, b), max(a, b)))
        return all_ar, frozenset(pairs)
    return False, frozenset()


def _form_is_aromatic(mol):
    """True if `mol` is the delocalized aromatic representative (a 6-ring all
    of whose bonds carry the aromatic flag). See `_aromatic_rep`."""
    return _aromatic_rep(mol)[0]


def _has_standard_kekule_ring(mol):
    """True if `mol` has a 6-membered ring with alternating single/double
    bonds (SDSDSD or DSDSDS) that is NOT the aromatic representative - i.e. a
    standard Kekulized benzene ring (S/D bonds with NO aromatic flag). This is
    RMG filtration's criterion for a 'redundant Kekule variant' of an aromatic
    form - the form that gets marked non-reactive (RMG
    mark_unreactive_structures). The aromatic representative itself (S/D WITH
    the aromatic flag, or 1.5) is NOT a standard Kekulized variant, so it must
    not be flagged (otherwise the aromatic form would be wrongly skipped).

    Computed on the STORED mol (`mol._rdkit`), not the AddHs graph (AddHs /
    sanitize re-aromatize a kekulized benzene ring, re-setting the flag).
    """
    try:
        em = mol._rdkit
    except Exception:
        return False
    for ring in em.GetRingInfo().AtomRings():
        if len(ring) != 6:
            continue
        orders = []
        all_ar = True
        for k in range(6):
            a, b = ring[k], ring[(k + 1) % 6]
            bd = em.GetBondBetweenAtoms(a, b)
            o = bd.GetBondTypeAsDouble()
            orders.append('D' if abs(o - 2) < 0.1 else 'S' if abs(o - 1) < 0.1 else '?')
            if not bd.GetIsAromatic():
                all_ar = False
        if all_ar:
            return False  # the aromatic representative, not a Kekulized variant
        return ''.join(orders) in ('SDSDSD', 'DSDSDS')
    return False


def _set_resonance_flags(structures):
    """Attach the two resonance-form flags RMG-Py carries on Molecule (the
    `reactive` flag + the delocalized-aromatic representation) to each form's
    RDKit mol, so the enumeration loop (Fix 2) and the group matcher (Fix 3)
    can read them.

    - `rmgpu_aromatic_rep` = 1: the form has the delocalized (aromatic, 1.5-bond)
      6-ring representation. The matcher reports its ring bonds as 1.5 (RMG
      compares benzene bonds as 1.5), so a `[S,D]`/`[D,T]` template does not
      match it while a `[D,T,B]` (benzene-including) template does.
    - `rmgpu_reactive` = 0: the form is a redundant Kekulized variant of an
      aromatic form (a standard SDSDSD/DSDSDS 6-ring while an aromatic form is
      also present). RMG's mark_unreactive_structures sets reactive=False on
      the filtered-out original (the Kekulized input) and the enumeration loop
      skips it - this is what stops the Kekulized benzylic form from
      generating the 3 spurious allene products.

    The flags live on the RDKit mol (not the Molecule wrapper) because the
    enumeration path rebuilds every form through the adjlist round-trip
    (assign_fresh_ids), which drops plain Python attributes but preserves RDKit
    mol properties that the caller re-applies.
    """
    has_aromatic = any(_form_is_aromatic(f) for f in structures)
    for f in structures:
        rdmol = f._rdkit
        if _form_is_aromatic(f):
            rdmol.SetProp('rmgpu_aromatic_rep', '1')
        elif rdmol.HasProp('rmgpu_aromatic_rep'):
            rdmol.DelProp('rmgpu_aromatic_rep')
        if not rdmol.HasProp('rmgpu_reactive'):
            rdmol.SetProp('rmgpu_reactive', '1')
        if (has_aromatic and not _form_is_aromatic(f)
                and _has_standard_kekule_ring(f)):
            rdmol.SetProp('rmgpu_reactive', '0')


def _allyl_bfs(seed_mols, max_depth=12):
    """Breadth-first expansion over the allyl radical delocalization shift,
    deduped by structural key. Returns the list of NEW forms reachable (beyond
    the seeds) by successive allyl shifts.

    The single-pass shift (the old behavior) reaches only ONE ortho form for a
    benzylic radical: from the kekulized input, shifting through the ipso
    double bond gives one valid ortho, but shifting through the ipso single
    aromatic bond is pentavalent and is dropped. Running the shift ITERATIVELY
    (RMG-Py's `_generate_resonance_structures` does exactly this: it applies the
    allyl method to every generated form) reaches the second ortho AND the para
    form, because from a proper cyclohexadienyl (ortho) form both further
    shifts are valid. This is what closes the job-05 Intra_ene gap (the para
    form is the sole source of product B, and the 2nd ortho is needed for A's
    degeneracy of 6 instead of 3).

    CRITICAL: the seed must be a KEKULIZED (explicit S/D) form, not the
    aromatic representative. A shift that converts an aromatic ring bond to a
    single bond leaves RDKit's Kekulize unable to resolve the ring (the
    remaining ring bonds are still aromatic-flagged and inconsistent), so the
    candidate is dropped and the BFS from an aromatic seed yields ZERO forms.
    From the kekulized input, the ipso double-bond shift is valid (one ortho),
    and from each ortho form the para shift is valid - reaching the full set.

    Dedup is by `_form_key` (the positional explicit-H bond-order signature),
    NOT the canonical SMILES: the two ortho forms share a canonical SMILES but
    differ in the radical POSITION, so a SMILES dedup would wrongly collapse
    them (RMG-Py keeps them as separate resonance forms).
    """
    seen = set()
    for s in seed_mols:
        seen.add(_form_key(s))
    frontier = [s._rdkit for s in seed_mols]
    out = []
    depth = 0
    while frontier and depth < max_depth:
        depth += 1
        nxt = []
        for rdmol in frontier:
            for s in _generate_allyl_delocalization_resonance_structures(rdmol):
                k = _form_key(s)
                if k not in seen:
                    seen.add(k)
                    out.append(s)
                    nxt.append(s._rdkit)
        frontier = nxt
    return out


def _get_lone_pairs(mol, atom):
    """Lone pairs per RMG-Py Molecule.update_lone_pairs():

        lone_pairs = (valence_electrons - radical_electrons - charge - bond_order) / 2

    Hydrogen (and lithium) always have 0. Matches rmgpu.molecule.adjlist.
    get_lone_pairs (the RMG-Py port).

    RMG-Py counts bond order on the EXPLICIT-H graph (Molecule.
    get_total_bond_order over its stored explicit-hydrogen vertices). rmgpu
    mols are stored implicit-H, so the bond order is computed after adding
    hydrogens. This is what keeps the lone-pair-radical generator from firing
    on pure hydrocarbon radicals: a [CH] radical carbon has total bond order
    4 once its implicit H is added, so LP = (4 - 1 - 0 - 4) < 0 -> 0, and no
    ionic resonance forms are emitted (RMG-Py never produces them). A per-
    element heuristic (the original) gave C/H nonzero lone pairs at all, which
    wrongly enabled the generator (a product-set divergence at the job-05 gate).
    """
    from rdkit import Chem
    from rmgpu.molecule.adjlist import VALENCE_ELECTRONS, _NO_LONE_PAIRS
    symbol = atom.GetSymbol()
    if symbol in _NO_LONE_PAIRS:
        return 0
    ve = VALENCE_ELECTRONS.get(symbol)
    if ve is None:
        return 0
    charge = atom.GetFormalCharge()
    unpaired = atom.GetNumRadicalElectrons()
    explicit = any(a.GetSymbol() == 'H' for a in mol.GetAtoms())
    if not explicit:
        try:
            em = Chem.AddHs(Chem.Mol(mol))
            eatom = em.GetAtomWithIdx(atom.GetIdx())
            bond_order = sum(b.GetBondTypeAsDouble() for b in eatom.GetBonds())
        except Exception:
            bond_order = sum(b.GetBondTypeAsDouble()
                             for b in atom.GetBonds())
    else:
        bond_order = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
    return max(0, (ve - unpaired - charge - int(bond_order)) // 2)


def _has_nitrogen_val5(mol):
    """Check for nitrogen valence 5 atoms without lone pairs."""
    return any(atom.GetSymbol() == 'N' and atom.GetFormalCharge() >= 1 for atom in mol.GetAtoms())


def _has_lone_pairs(mol):
    """Check if molecule has any atoms with lone pairs."""
    return any(_get_lone_pairs(mol, atom) > 0 for atom in mol.GetAtoms())


def _is_aromatic(mol):
    """Check if molecule is aromatic by detecting aromatic bonds in rings."""
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            return True
    return False


def _is_cyclic(mol):
    """Check if molecule has a ring."""
    return any(len(ring) >= 3 for ring in mol.GetRingInfo().AtomRings())


def _analyze_molecule(mol):
    """Analyze molecule features for resonance generation."""
    return {
        'is_radical': _is_radical(mol),
        'is_cyclic': _is_cyclic(mol),
        'is_aromatic': _is_aromatic(mol),
        'isPolycyclicAromatic': False,  # Simplified for now
        'is_aryl_radical': False,       # Simplified for now
        'hasNitrogenVal5': _has_nitrogen_val5(mol),
        'hasLonePairs': _has_lone_pairs(mol),
        'is_multidentate': False        # Simplified for now
    }


def _generate_allyl_delocalization_resonance_structures(mol):
    """Generate resonance structures by allyl radical shift.

    A radical adjacent to a pi bond delocalizes: the radical moves to the far
    end of the pi bond and the pi bond becomes a single bond (the new pi bond
    forms between the original radical site and the near end). The pi bond is
    matched as an explicit DOUBLE bond OR an AROMATIC ring bond, so a radical
    exocyclic to an aromatic ring (a benzylic radical) also delocalizes into
    the ring, forming an exocyclic C=C and a radical on the ring - exactly
    the resonance forms RMG-Py generates for e.g. 1-phenylethyl.

    The radical atom itself must be NON-aromatic: shifting within an aromatic
    ring (an aryl radical) would break the ring's pi system, and RMG-Py does
    not generate those (the ring is already the delocalized form).
    """
    structures = []
    if not _is_radical(mol):
        return structures

    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() == 0:
            continue
        # An aryl radical (radical on an aromatic ring atom) is already the
        # delocalized form - do not shift within the ring.
        if atom.GetIsAromatic():
            continue

        rad_idx = atom.GetIdx()

        # The pi bond the radical delocalizes through must have the radical
        # bonded directly to one of its two atoms. In a kekulized aromatic
        # ring the ipso carbon (the ring atom the radical is bonded to) carries
        # ONE double and ONE single ring bond, but both are "pi bonds" in the
        # aromatic resonance - the radical delocalizes to BOTH ortho carbons.
        # So include every pi bond (explicit double or aromatic) adjacent to
        # the radical, PLUS the single ring bond of an aromatic ipso carbon.
        candidates = {}
        for b2 in atom.GetBonds():
            other_idx = b2.GetOtherAtomIdx(rad_idx)
            neighbor_atom = mol.GetAtomWithIdx(other_idx)
            for bond in neighbor_atom.GetBonds():
                if bond.GetBeginAtomIdx() == rad_idx or \
                        bond.GetEndAtomIdx() == rad_idx:
                    continue  # the C(radical)-C(bonded) bond itself
                bt = bond.GetBondType()
                is_pi = bt in (BondType.DOUBLE, BondType.AROMATIC)
                # A single ring bond of an aromatic ipso carbon is a valid
                # delocalization path (the other ortho resonance form).
                is_aromatic_ring_bond = (
                    not is_pi and bond.GetIsAromatic() and
                    neighbor_atom.GetIsAromatic())
                if is_pi or is_aromatic_ring_bond:
                    a1 = bond.GetBeginAtomIdx()
                    a2 = bond.GetEndAtomIdx()
                    # The bond must connect the radical to the far atom:
                    # one endpoint is the ipso (bonded) carbon, the other is
                    # the far atom the radical migrates to.
                    if other_idx in (a1, a2):
                        key = frozenset((a1, a2))
                        if key not in candidates:
                            candidates[key] = bond

        for bond in candidates.values():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()

            # Determine target (the atom opposite the radical on the pi bond).
            # The radical is NOT an endpoint of this bond (it's bonded to the
            # ipso carbon which IS an endpoint); identify which endpoint is
            # bonded to the radical.
            rad_bonded_endpoint = None
            for b2 in atom.GetBonds():
                o = b2.GetOtherAtomIdx(rad_idx)
                if o == a1 or o == a2:
                    rad_bonded_endpoint = o
                    break
            if rad_bonded_endpoint is None:
                continue
            if rad_bonded_endpoint == a1:
                target_idx = a2
                near_idx = a1
            else:
                target_idx = a1
                near_idx = a2

            # Create new structure with shifted radical and bonds.
            new_mol = Chem.RWMol(mol)

            # Move radical to target.
            new_mol.GetAtomWithIdx(target_idx).SetNumRadicalElectrons(
                new_mol.GetAtomWithIdx(target_idx).GetNumRadicalElectrons() + 1)
            new_mol.GetAtomWithIdx(rad_idx).SetNumRadicalElectrons(0)

            # The pi bond becomes single; a new pi bond forms between the
            # original radical site and the near end (forming the exocyclic
            # C=C when the pi bond was an aromatic ring bond).
            new_mol.GetBondBetweenAtoms(a1, a2).SetBondType(BondType.SINGLE)
            new_mol.GetBondBetweenAtoms(rad_idx, near_idx).SetBondType(
                BondType.DOUBLE)

            # Clear aromaticity on the touched bonds so kekulization sees the
            # explicit bond orders (the ring is now a kekulized ring bearing
            # an exocyclic double bond).
            for b in new_mol.GetBonds():
                b.SetIsAromatic(False)

            # Validate the structure. The radical landing on a ring carbon
            # leaves a stale valence cache, so sanitize first (otherwise
            # Kekulize reports "Unkekulized atoms" even for a valid shift);
            # an invalid shift - e.g. the radical landing on a fully-
            # substituted ring atom - still raises KekulizeException and the
            # candidate is dropped. This rdkit build's Kekulize returns None
            # (not True) on success, so a raised exception is the only failure
            # signal.
            try:
                Chem.SanitizeMol(new_mol)
                Chem.Kekulize(new_mol, clearAromaticFlags=True)
            except Exception:
                pass
            else:
                structures.append(Molecule._from_rdmol(Chem.Mol(new_mol)))
    return structures


def _generate_lone_pair_multiple_bond_resonance_structures(mol):
    """Generate resonance structures by lone pair shift with double/triple bond."""
    structures = []
    if not _has_lone_pairs(mol):
        return structures
        
    # Find 3-atom systems where lone pair can shift to multiple bond
    for atom in mol.GetAtoms():
        if _get_lone_pairs(mol, atom) == 0:
            continue
            
        # Look for adjacent atoms with multiple bonds
        for bond in atom.GetBonds():
            if bond.GetBondType() in [BondType.DOUBLE, BondType.TRIPLE]:
                other_idx = bond.GetOtherAtomIdx(atom.GetIdx())
                # Check if other atom has another multiple bond (conjugated system)
                for bond2 in mol.GetAtomWithIdx(other_idx).GetBonds():
                    if bond2.GetBondType() in [BondType.DOUBLE, BondType.TRIPLE] and bond2 != bond:
                        third_idx = bond2.GetOtherAtomIdx(other_idx)
                        third_lone_pairs = _get_lone_pairs(mol, mol.GetAtomWithIdx(third_idx))
                        
                        # Create new structure with shifted lone pair and bond orders
                        new_mol = Chem.RWMol(mol)
                        # Just adjust bond orders and charges for now, skip lone pair manipulation
                        new_mol.GetBondBetweenAtoms(atom.GetIdx(), other_idx).SetBondType(BondType.DOUBLE)
                        new_mol.GetBondBetweenAtoms(other_idx, third_idx).SetBondType(BondType.SINGLE)
                        
                        # Update formal charges
                        new_mol.GetAtomWithIdx(atom.GetIdx()).SetFormalCharge(
                            new_mol.GetAtomWithIdx(atom.GetIdx()).GetFormalCharge() + 1)
                        new_mol.GetAtomWithIdx(third_idx).SetFormalCharge(
                            new_mol.GetAtomWithIdx(third_idx).GetFormalCharge() - 1)
                        
                        if Chem.Kekulize(new_mol):
                            structures.append(Molecule._from_rdmol(Chem.Mol(new_mol)))
    return structures


def _generate_adj_lone_pair_radical_resonance_structures(mol):
    """Generate resonance structures by adjacent lone pair-radical shift."""
    structures = []
    if not _is_radical(mol) or not _has_lone_pairs(mol):
        return structures
        
    # Find adjacent atoms where one has radical, other has lone pair
    for bond in mol.GetBonds():
        a1_idx = bond.GetBeginAtomIdx()
        a2_idx = bond.GetEndAtomIdx()
        
        a1 = mol.GetAtomWithIdx(a1_idx)
        a2 = mol.GetAtomWithIdx(a2_idx)
        
        # Check if a1 has radical and a2 has lone pair, or vice versa
        if (a1.GetNumRadicalElectrons() > 0 and _get_lone_pairs(mol, a2) > 0) or \
           (a2.GetNumRadicalElectrons() > 0 and _get_lone_pairs(mol, a1) > 0):
            
            # Determine direction
            if a1.GetNumRadicalElectrons() > 0 and _get_lone_pairs(mol, a2) > 0:
                rad_idx, lp_idx = a1_idx, a2_idx
            else:
                rad_idx, lp_idx = a2_idx, a1_idx
            
            # Create new structure
            new_mol = Chem.RWMol(mol)

            # Move the radical from the radical site to the lone-pair site.
            # The radical site gains a lone pair (charge decreases by 1) and
            # the lone-pair site gains the radical (charge increases by 1).
            # This mirrors RMG-Py's adj lone-pair-radical resonance transition.
            new_mol.GetAtomWithIdx(rad_idx).SetNumRadicalElectrons(0)
            new_mol.GetAtomWithIdx(lp_idx).SetNumRadicalElectrons(
                new_mol.GetAtomWithIdx(lp_idx).GetNumRadicalElectrons() + 1)

            # Update formal charges (rad site -1, lone-pair site +1)
            new_mol.GetAtomWithIdx(rad_idx).SetFormalCharge(
                new_mol.GetAtomWithIdx(rad_idx).GetFormalCharge() - 1)
            new_mol.GetAtomWithIdx(lp_idx).SetFormalCharge(
                new_mol.GetAtomWithIdx(lp_idx).GetFormalCharge() + 1)

            try:
                new_mol.UpdatePropertyCache()
                Chem.SanitizeMol(new_mol)
            except Exception:
                continue
            structures.append(Molecule._from_rdmol(Chem.Mol(new_mol)))
    return structures


def _generate_optimal_aromatic_resonance_structures(mol):
    """Generate optimal aromatic resonance structures for aromatic molecules."""
    if not _is_aromatic(mol):
        return []
        
    # Convert to aromatic form
    new_mol = Chem.RWMol(mol)
    try:
        Chem.Kekulize(new_mol, clearAromaticFlags=False)
        Chem.SetAromaticity(new_mol)
        return [Molecule._from_rdmol(Chem.Mol(new_mol))]
    except Exception:
        return []


def _generate_kekule_structure(mol):
    """Generate kekulized structure for aromatic molecules."""
    if not _is_aromatic(mol):
        return []
        
    try:
        new_mol = Chem.RWMol(mol)
        Chem.Kekulize(new_mol, clearAromaticFlags=True)
        return [Molecule._from_rdmol(Chem.Mol(new_mol))]
    except Exception:
        return []


def generate_resonance_structures(molecule: Molecule) -> list[Molecule]:
    """
    Generate resonance structures for a molecule using RMG-style rules.

    Args:
        molecule: A Molecule instance for which to generate resonance structures.

    Returns:
        A list of Molecule instances representing the resonance structures.
        The input molecule is included in the list.

    The benzylic / aromatic-radical handling (job-05 step-06): a radical
    exocyclic to a benzene ring delocalizes to BOTH ortho positions and the
    para position, in addition to the delocalized aromatic form and the
    Kekulized input form - the 5-form set RMG-Py generates for 1-phenylethyl.
    The single-pass allyl shift (the old behavior) reaches only ONE ortho
    form for the kekulized input, so the allyl generator is now run
    ITERATIVELY (BFS, mirroring RMG-Py's `_generate_resonance_structures`
    which applies each method to every generated form) and seeded from the
    KEKULED input (the aromatic representative CANNOT seed the shift: a shift
    off an aromatic ring bond leaves RDKit's kekulizer unable to resolve the
    ring, so all candidates are dropped). From the kekulized input the ipso
    double-bond shift gives one ortho form, and from each ortho form the para
    shift is valid.
    The forms are deduped by their STRUCTURAL key (the exact explicit-H
    bond-order signature with 1.5 kept distinct from S/D, plus per-atom
    radical/charge), NOT the canonical SMILES: the two ortho benzylic forms
    share a canonical SMILES but differ in radical position, and the aromatic
    representative and the kekulized form also collapse to one SMILES - all
    three distinctions are what RMG-Py keeps (5 distinct forms). Finally the
    two RMG-Py resonance-form flags are attached to each form's RDKit mol
    (`_set_resonance_flags`): the aromatic representative (matcher reports
    its 6-ring bonds as 1.5) and the redundant Kekulized variant (marked
    non-reactive, skipped by the enumeration loop).
    """
    # Get RDKit molecule and analyze features
    rdmol = molecule._rdkit
    features = _analyze_molecule(rdmol)

    new_structures = []

    # Allyl radical delocalization (the benzylic forms). The radical atom is
    # guarded inside the generator to be non-aromatic, so in-ring (aryl)
    # radicals do not break their ring's pi system, while exocyclic radicals
    # (a benzylic radical) delocalize into the ring. It is run ITERATIVELY
    # (BFS) so both ortho forms AND the para form are reached. The seed is the
    # KEKULED input (never the aromatic representative - see _allyl_bfs).
    if features['is_radical']:
        new_structures.extend(_allyl_bfs([molecule.copy()]))

    if features['hasLonePairs'] and not features['is_aromatic']:
        new_structures.extend(_generate_lone_pair_multiple_bond_resonance_structures(rdmol))
        new_structures.extend(_generate_adj_lone_pair_radical_resonance_structures(rdmol))

    if features['is_aromatic']:
        new_structures.extend(_generate_optimal_aromatic_resonance_structures(rdmol))
        new_structures.extend(_generate_kekule_structure(rdmol))

    # Deduplicate by the STRUCTURAL key (exact explicit-H bond-order
    # signature), NOT the canonical SMILES: the two ortho benzylic forms
    # share a canonical SMILES but differ in radical position, and the
    # aromatic representative + kekulized form share a SMILES too - all
    # must be kept (RMG-Py carries 5 distinct forms). The input form is
    # always preserved first.
    all_structures = [molecule.copy()]
    seen = {_form_key(all_structures[0])}
    for structure in new_structures:
        k = _form_key(structure)
        if k not in seen:
            seen.add(k)
            all_structures.append(structure)

    # Attach the RMG-Py resonance-form flags (aromatic-representative +
    # reactive) to each form's RDKit mol (Fix 2 / Fix 3 read these).
    _set_resonance_flags(all_structures)

    # Filtration: keep only representative structures (RMG-Py
    # filtration.filter_structures), always preserving the input.
    from rmgpu.molecule.resonance_filtration import filter_structures
    result = filter_structures(all_structures, molecule.copy())

    # Filtration's octet/charge pass can drop forms; re-assert the flags on
    # the survivors so the enumeration/matcher see the correct flags.
    _set_resonance_flags(result)
    return result


# Expose this function as a method on Molecule
Molecule.get_resonance_structures = generate_resonance_structures
