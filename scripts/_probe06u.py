"""Probe u: end-to-end fix-3 verification with the REAL pipeline.
1. Build the 5-form benzylic set (kek-seed BFS + optimal-aromatic + kekule input).
2. Capture per-form: is_aromatic_rep (all ring bonds 1.5+flag) + ring bond pairs.
3. Run the real assign_fresh_ids round-trip.
4. Re-aromatize the aromatic-rep form's ring bonds.
5. Count Intra_ene template matchings per form via the REAL matcher path
   (TemplateMatcher.match_molecule) and check the expected pattern:
   aromatic -> 0, ortho1/ortho2/para -> 4/6/5-ish (nonzero), kek -> nonzero.
6. Also verify step-03 is untouched: kekulized benzene (C1=CC=CC=C1) forms
   still match R_Addition_MultipleBond as before.
"""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rdkit.Chem import BondType
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import (
    _allyl_bfs, _form_key, _generate_optimal_aromatic_resonance_structures,
    _generate_kekule_structure, _ring_6_orders)
from rmgpu.core.family import KineticsFamilies
from rmgpu.core.template import TemplateMatcher
from rmgpu.core import enumeration as enum


def form_signature(mol):
    """Exact explicit-H bond-order signature (1.5 distinct from S/D)."""
    em = mol._with_explicit_h()
    sig = []
    for b in em.GetBonds():
        if b.GetBondTypeAsDouble() in (1.0, 1.5, 2.0, 3.0):
            o = b.GetBondTypeAsDouble()
            sig.append((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                        max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), round(o, 2)))
    return tuple(sorted(sig))


def aromatic_rep_info(mol):
    """Return (is_aromatic_rep, ring_bond_pairs) using explicit-H ring."""
    em = mol._with_explicit_h()
    ri = em.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) != 6:
            continue
        pairs = set()
        allar = True
        for k in range(6):
            a, b = ring[k], ring[(k + 1) % 6]
            bd = em.GetBondBetweenAtoms(a, b)
            if not (bd.GetIsAromatic() and abs(bd.GetBondTypeAsDouble() - 1.5) < 0.1):
                allar = False
                break
            pairs.add((min(a, b), max(a, b)))
        if allar:
            return True, frozenset(pairs)
        return False, frozenset()
    return False, frozenset()


mol = Molecule(smiles='C[CH]C1=CC=CC=C1')

# 1. Build the 5-form set with the CORRECT dedup (exact signature)
cands = [mol.copy()]
cands.extend(_allyl_bfs([mol.copy()]))  # kek seed
cands.extend(_generate_optimal_aromatic_resonance_structures(mol._rdkit))
cands.extend(_generate_kekule_structure(mol._rdkit))
seen = set()
forms = []
for f in cands:
    s = form_signature(f)
    if s not in seen:
        seen.add(s)
        forms.append(f)
print('form set (exact-signature dedup):', len(forms))
captured = []
for i, f in enumerate(forms):
    is_ar, pairs = aromatic_rep_info(f)
    pat, all_ar = _ring_6_orders(f)
    captured.append((f, is_ar, pairs))
    print('  [%d] %s ar_rep=%s ring_pairs=%s' % (
        i, f.to_smiles(), is_ar, sorted(pairs)))

# 2. Run real assign_fresh_ids
species = [list(forms)]
enum.assign_fresh_ids(species)

# 3. Re-aromatize aromatic-rep form
reapplied = []
for f, is_ar, pairs in captured:
    rt = species[0][list(forms).index(f)]  # same order
    if is_ar:
        em = rt._with_explicit_h()
        rw = Chem.RWMol(em)
        for (a, b) in pairs:
            bd = rw.GetBondBetweenAtoms(a, b)
            if bd is not None:
                bd.SetBondType(BondType.AROMATIC)
        for b in rw.GetBonds():
            if b.GetBondType() != BondType.AROMATIC:
                b.SetIsAromatic(False)
        Chem.SanitizeMol(rw)
        # re-kekulize check: can it?
        try:
            Chem.Kekulize(rw, clearAromaticFlags=False)
        except Exception as e:
            print('  kekulize re-arom failed:', e)
        rt = Molecule._from_rdmol(Chem.Mol(rw))
        # re-attach atomid props? they were on the old mols; redo minimal
    is_ar2, _ = aromatic_rep_info(rt)
    print('  post-rt ar_rep form: is_arom_rep_now=%s smiles=%s' % (is_ar2, rt.to_smiles()))
    reapplied.append(rt)

# 4. Intra_ene matching per form (real matcher)
kf = KineticsFamilies().load('default')
fam = {f.label: f for f in kf.families}['Intra_ene_reaction']
matcher = TemplateMatcher(fam)
efam = enum.Family(
    label=fam.label, recipe=fam.recipe, reverse_recipe=fam.reverse_recipe,
    own_reverse=fam.own_reverse, reversible=fam.reversible,
    allow_charged_species=fam.allow_charged_species, electrons=fam.electrons,
    reactant_num_effective=fam.num_template_reactants_effective,
    product_num_forward=fam.product_num_forward, reverse_map=fam.reverse_map,
    template_labels=[e.label for e in fam.forward_template], forbidden=fam.forbidden,
)
efam.matcher = matcher
print('\nIntra_ene matchings per form (re-aromatized aromatic rep):')
for i, f in enumerate(reapplied):
    ms = matcher.match_molecule(f, 0, 'ab')
    print('  form %d: %d matchings' % (i, len(ms)))

# full enumeration
rxns = enum.generate_reactions(efam, [mol], matcher=matcher)


def canon(p):
    r2 = p._rdkit
    if not any(a.GetSymbol() == 'H' for a in r2.GetAtoms()):
        try:
            r2 = Chem.AddHs(r2)
        except Exception:
            pass
    try:
        Chem.Kekulize(r2, clearAromaticFlags=True)
    except Exception:
        pass
    return Chem.MolToSmiles(r2)


print('\nIntra_ene full enumeration (5 forms, re-aro rep, NO reactive skip yet):')
for r in rxns:
    print('  deg=%.1f products=%s' % (r.degeneracy, [canon(p) for p in r.products]))
