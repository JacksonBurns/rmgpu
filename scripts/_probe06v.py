"""Probe v: CLEAN end-to-end. No index games.
Build 5-form set (kek seed), keep an explicit parallel list of (form, ar_rep, pairs),
run assign_fresh_ids, re-aromatize ONLY the ar_rep form, verify each form's
matcher bond orders, count Intra_ene matchings, and run the gate case.
"""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rdkit.Chem import BondType
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import (
    _allyl_bfs, _generate_optimal_aromatic_resonance_structures,
    _generate_kekule_structure)


def sig(mol):
    em = mol._with_explicit_h()
    s = []
    for b in em.GetBonds():
        o = b.GetBondTypeAsDouble()
        if o in (1.0, 1.5, 2.0, 3.0):
            s.append((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                      max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), round(o, 2)))
    return tuple(sorted(s))


def ar_rep(mol):
    """(is_aromatic_rep, ring_pairs) - all 6 ring bonds aromatic AND 1.5."""
    em = mol._with_explicit_h()
    for ring in em.GetRingInfo().AtomRings():
        if len(ring) != 6:
            continue
        pairs, ok = set(), True
        for k in range(6):
            a, b = ring[k], ring[(k + 1) % 6]
            bd = em.GetBondBetweenAtoms(a, b)
            if not (bd.GetIsAromatic() and abs(bd.GetBondTypeAsDouble() - 1.5) < 0.1):
                ok = False
                break
            pairs.add((min(a, b), max(a, b)))
        return ok, frozenset(pairs)
    return False, frozenset()


def bond_orders(mol):
    """sorted (pair, order, is_aromatic) for the 6-ring, explicit-H."""
    em = mol._with_explicit_h()
    out = []
    for ring in em.GetRingInfo().AtomRings():
        if len(ring) == 6:
            for k in range(6):
                a, b = ring[k], ring[(k + 1) % 6]
                bd = em.GetBondBetweenAtoms(a, b)
                out.append((min(a, b), max(a, b), round(bd.GetBondTypeAsDouble(), 2), bd.GetIsAromatic()))
            break
    return sorted(out)


mol = Molecule(smiles='C[CH]C1=CC=CC=C1')
cands = [mol.copy()]
cands.extend(_allyl_bfs([mol.copy()]))
cands.extend(_generate_optimal_aromatic_resonance_structures(mol._rdkit))
cands.extend(_generate_kekule_structure(mol._rdkit))
seen, forms = set(), []
for f in cands:
    s = sig(f)
    if s not in seen:
        seen.add(s); forms.append(f)
print('== form set (exact-sig dedup):', len(forms))
meta = []
for i, f in enumerate(forms):
    a, p = ar_rep(f)
    meta.append((a, p))
    print('  [%d] %s  ar_rep=%s  ring=%s' % (i, f.to_smiles(), a, bond_orders(f)))

# assign_fresh_ids (round-trip)
from rmgpu.core import enumeration as enum
species = [list(forms)]
enum.assign_fresh_ids(species)
print('\n== post assign_fresh_ids (round-trip):')
post = []
for i, f in enumerate(species[0]):
    a, p = ar_rep(f)
    post.append((a, p))
    print('  [%d] %s  ar_rep_now=%s  ring=%s' % (i, f.to_smiles(), a, bond_orders(f)))

# re-aromatize the ar_rep form (the one that was ar_rep pre-rt)
reapplied = list(species[0])
for i, (a, p) in enumerate(meta):
    if not a:
        continue
    f = species[0][i]
    em = f._with_explicit_h()
    rw = Chem.RWMol(em)
    for (aa, bb) in p:
        bd = rw.GetBondBetweenAtoms(aa, bb)
        if bd is not None:
            bd.SetBondType(BondType.AROMATIC)
    for b in rw.GetBonds():
        if b.GetBondType() != BondType.AROMATIC:
            b.SetIsAromatic(False)
    Chem.SanitizeMol(rw)
    new = Molecule._from_rdmol(Chem.Mol(rw))
    print('\n  RE-AROM form [%d]: %s  ar_rep_now=%s  ring=%s' % (
        i, new.to_smiles(), ar_rep(new)[0], bond_orders(new)))
    reapplied[i] = new

# Intra_ene matchings
from rmgpu.core.family import KineticsFamilies
from rmgpu.core.template import TemplateMatcher
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
print('\n== Intra_ene matchings per re-aromatized form:')
for i, f in enumerate(reapplied):
    ms = matcher.match_molecule(f, 0, 'ab')
    print('  form %d: %d matchings' % (i, len(ms)))

def canon(p):
    r2 = p._rdkit
    if not any(a.GetSymbol() == 'H' for a in r2.GetAtoms()):
        try: r2 = Chem.AddHs(r2)
        except Exception: pass
    try: Chem.Kekulize(r2, clearAromaticFlags=True)
    except Exception: pass
    return Chem.MolToSmiles(r2)

# The gate compares canonical product sets; A = C=CC1=CCC=C[CH]1, B = C=CC1=CC[CH]C=C1
rxns = enum.generate_reactions(efam, [mol], matcher=matcher)
print('\n== Intra_ene full enumeration (re-aro rep, NO reactive-skip):')
for r in rxns:
    print('  deg=%.1f products=%s' % (r.degeneracy, [canon(p) for p in r.products]))
