"""Probe x: verify flag-gated 1.5 in the matcher zeroes the aromatic form's
Intra_ene matchings (the clean fix-3), while leaving kekulized/ortho/para forms'
matchings intact. Simulates the _explicit_graph gate (report 1.5 for 6-ring
bonds of a flagged form) by building the graph manually + match_explicit_graph.
"""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import (
    _allyl_bfs, _generate_optimal_aromatic_resonance_structures,
    _generate_kekule_structure)
from rmgpu.molecule.group import match_explicit_graph, _atom_matches_group_atom
from rmgpu.core.family import KineticsFamilies
from rmgpu.core.template import TemplateMatcher
from rmgpu.core import enumeration as enum
from rmgpu.molecule.atomtype import assign_atom_types
from rmgpu.molecule.adjlist import get_lone_pairs


def gated_explicit_graph(mol, gate=False):
    """_explicit_graph but, if gate and mol has rmgpu_aromatic_rep, report
    6-ring bonds as 1.5 (the fix-3 matcher change)."""
    em = mol._with_explicit_h()
    n = em.GetNumAtoms()
    types = assign_atom_types(mol)
    atoms = []
    for i, a in enumerate(em.GetAtoms()):
        atoms.append({'symbol': a.GetSymbol(), 'radical': a.GetNumRadicalElectrons(),
                      'charge': a.GetFormalCharge(), 'lone_pairs': None,
                      'atomtype': types[i] if i < len(types) else None})
    # ring bond set (6-membered rings)
    ring_bonds = set()
    if gate:
        for ring in em.GetRingInfo().AtomRings():
            if len(ring) == 6:
                for k in range(6):
                    a, b = ring[k], ring[(k + 1) % 6]
                    ring_bonds.add((min(a, b), max(a, b)))
    adj = [[] for _ in range(n)]
    for b in em.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        o = b.GetBondTypeAsDouble()
        if gate and (min(i, j), max(i, j)) in ring_bonds:
            o = 1.5
        adj[i].append((j, o))
        adj[j].append((i, o))
    for i, f in enumerate(atoms):
        bo = sum(o for (_j, o) in adj[i])
        try:
            f['lone_pairs'] = get_lone_pairs(f['symbol'], f['radical'], f['charge'], bo)
        except Exception:
            f['lone_pairs'] = None
    return {'atoms': atoms, 'adj': adj}


mol = Molecule(smiles='C[CH]C1=CC=CC=C1')
cands = [mol.copy()]
cands.extend(_allyl_bfs([mol.copy()]))
cands.extend(_generate_optimal_aromatic_resonance_structures(mol._rdkit))
cands.extend(_generate_kekule_structure(mol._rdkit))
# exact-signature dedup
def sig(m):
    em = m._with_explicit_h()
    s = set()
    for b in em.GetBonds():
        o = b.GetBondTypeAsDouble()
        if o in (1, 1.5, 2, 3):
            s.add((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), round(o, 2)))
    return frozenset(s)
seen, forms = set(), []
for f in cands:
    s = sig(f)
    if s not in seen:
        seen.add(s); forms.append(f)
print('form set:', len(forms))

# which is the aromatic-rep (all 6 ring bonds 1.5)?
def is_arom_rep(m):
    em = m._with_explicit_h()
    for ring in em.GetRingInfo().AtomRings():
        if len(ring) == 6:
            ok = all(abs(em.GetBondBetweenAtoms(ring[k], ring[(k+1)%6]).GetBondTypeAsDouble()-1.5) < 0.1 for k in range(6))
            return ok
    return False

# run round-trip
species = [list(forms)]
enum.assign_fresh_ids(species)
rt = species[0]

# Get the Intra_ene template groups
kf = KineticsFamilies().load('default')
fam = {f.label: f for f in kf.families}['Intra_ene_reaction']
matcher = TemplateMatcher(fam)

print('\nPer form: matchings WITHOUT gate (current) vs WITH flag-gate:')
for i, f in enumerate(rt):
    ar = is_arom_rep(f)  # pre-rt form (forms[i]) has the 1.5 ring
    # WITHOUT gate
    g_nogate = gated_explicit_graph(f, gate=False)
    # build the forward template group for slot 0
    grp = fam.forward_template[0]
    # match via match_explicit_graph with the group's atoms/edges
    # Use the matcher's internal group for slot 0
    m_nogate = match_explicit_graph(g_nogate, grp)
    # WITH gate (only meaningful if this form is the arom rep)
    if ar:
        g_gate = gated_explicit_graph(f, gate=True)
        m_gate = match_explicit_graph(g_gate, grp)
        print('  form %d ar_rep=%s: nogate=%d  GATE=%d' % (i, ar, len(m_nogate), len(m_gate)))
    else:
        print('  form %d ar_rep=%s: nogate=%d  (no gate)' % (i, ar, len(m_nogate)))
