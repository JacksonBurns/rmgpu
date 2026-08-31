#!/usr/bin/env python3
"""
job-05/step-03 reference capture.

Records, from RMG-Py (rmg_env, never rmgpu), the ground truth for the
template-matching / group-matcher port:

  1. The full atom-type tree (rmgpy.molecule.atomtype.ATOMTYPES): for every
     atom type its label, generic-type labels and specific-type labels. This
     is what GroupAtom.is_specific_case_of / equivalent / AtomType relations
     depend on; recording it (rather than re-deriving) keeps the port faithful
     and non-circular.

  2. For each of the 7 mechanism families (the ones that drive the step-02
     product-enumeration set): the full kinetics group TREE - every entry's
     label, its group adjacency list (RMG's own to_adjacency_list), its parent
     label, its children labels, and whether the item is a LogicNode (OR/AND)
     with which components. Plus the forward/reverse template entries and their
     forward-template reactant labels.

  3. For each step-02 (family, reactants) case, RMG's own matching results on
     the base (non-resonance) reactant molecules:
       - _match_reactant_to_template(reactant, template_slot): the number of
         subgraph-isomorphism mappings and the normalized set of mappings
         (each mapping = sorted list of (reactant_atom_index, template_label));
       - get_reaction_template_labels(reaction): the MOST-SPECIFIC template
         labels RMG's tree walk returns (the 'template' a reaction carries).
     The normalized-mapping representation is stable across the two
     implementations, so the rmgpu port is checked against it directly.

  4. The group-construct inventory: which RMG-specific constructs actually
     appear in the recorded group adjacency lists (multi-atomtype {a,b},
     wildcard u[x]/p[x]/c[x], multi-order bonds {S,D}, benzene bonds, inRing
     props, multiplicity lines). Drives the port-vs-delegate decisions.

Non-circular: uses only rmgpy + the RMG databases, never rmgpu code.

Run with: /home/jackson/miniforge3/envs/rmg_env/bin/python scripts/record_job05_step03_reference.py
"""
import json
import os
import re

from rmgpy import settings
from rmgpy.data.base import LogicNode
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.molecule import Molecule
from rmgpy.molecule.atomtype import ATOMTYPES
from rmgpy.reaction import Reaction

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAMILIES = {
    'real': ['H_Abstraction', 'R_Addition_MultipleBond',
             'intra_H_migration', 'Intra_ene_reaction'],
    'test': ['R_Recombination', '1,2_shiftC', 'Singlet_Val6_to_triplet'],
}

# The step-02 (family, source, reactant SMILES) case set - the reactions whose
# generated products must round-trip-match their family template.
CASES = [
    ('H_Abstraction', 'real', ['C', '[H]']),
    ('H_Abstraction', 'real', ['CC', '[OH]']),
    ('H_Abstraction', 'real', ['CC(=O)C', '[H]']),
    ('H_Abstraction', 'real', ['C[CH]C', '[H]']),
    ('H_Abstraction', 'real', ['C1=CC=CC=C1', '[H]']),
    ('H_Abstraction', 'real', ['C=C', '[CH3]']),
    ('H_Abstraction', 'real', ['C(C)(C)C', '[CH3]']),
    ('H_Abstraction', 'real', ['CC(=O)[O]', '[H]']),
    ('H_Abstraction', 'real', ['C1CCC1', '[OH]']),
    ('H_Abstraction', 'real', ['C1CC1', '[CH3]']),
    ('H_Abstraction', 'real', ['CCO', '[OH]']),
    ('H_Abstraction', 'real', ['C#C', '[H]']),
    ('H_Abstraction', 'real', ['CCN', '[H]']),
    ('H_Abstraction', 'real', ['CC#C', '[CH3]']),
    ('R_Recombination', 'test', ['[OH]', '[OH]']),
    ('R_Recombination', 'test', ['[CH3]', '[OH]']),
    ('R_Recombination', 'test', ['[CH3]', '[CH3]']),
    ('R_Addition_MultipleBond', 'real', ['C=C', '[CH3]']),
    ('R_Addition_MultipleBond', 'real', ['C=C', '[OH]']),
    ('R_Addition_MultipleBond', 'real', ['C1=CC=CC=C1', '[H]']),
    ('R_Addition_MultipleBond', 'real', ['C=O', '[CH3]']),
    ('R_Addition_MultipleBond', 'real', ['C#C', '[H]']),
    ('R_Addition_MultipleBond', 'real', ['C#C', '[CH3]']),
    ('intra_H_migration', 'real', ['C[CH]CCC']),
    ('intra_H_migration', 'real', ['C[CH]C1CCCCC1']),
    ('intra_H_migration', 'real', ['C(C)(C)[CH]C(C)(C)C']),
    ('Intra_ene_reaction', 'real', ['C[CH]C1=CC=CC=C1']),
    ('1,2_shiftC', 'test', ['CC[CH]C']),
    ('1,2_shiftC', 'test', ['CCC[CH]C(C)C']),
    ('Singlet_Val6_to_triplet', 'test', ['O=O']),
]

CONSTRUCT_RE = {
    'multi_atomtype': re.compile(r'\{[A-Za-z0-9!]+,[^}]+\}'),
    'wildcard_u': re.compile(r'u\[x\]'),
    'wildcard_p': re.compile(r'p\[x\]'),
    'wildcard_c': re.compile(r'c\[x\]'),
    'multi_order_bond': re.compile(r'\{[SDTBQ],\[?'),
    'benzene_bond': re.compile(r',B\}'),
    'inring': re.compile(r'\br[01]\b'),
    'multiplicity': re.compile(r'^\s*multiplicity\s+\['),
}


def _atomtype_tree():
    tree = {}
    for label, at in ATOMTYPES.items():
        tree[label] = {
            'generic': [g.label for g in at.generic],
            'specific': [s.label for s in at.specific],
        }
    return tree


def _group_adj(group):
    return group.to_adjacency_list()


def _record_family(fam):
    g = fam.groups
    tree = {}
    for entry in g.entries.values():
        item = entry.item
        if isinstance(item, LogicNode):
            node = {
                'type': 'logic',
                'symbol': getattr(item, 'symbol', None),
                'invert': bool(item.invert),
                'components': [getattr(c, 'label', c) for c in item.components],
            }
        else:
            node = {
                'type': 'group',
                'adj': _group_adj(item),
            }
        tree[entry.label] = {
            'parent': entry.parent.label if entry.parent is not None else None,
            'children': [c.label for c in entry.children],
            'node': node,
        }
    fwd = [e.label for e in fam.forward_template.reactants]
    rev = ([e.label for e in fam.reverse_template.reactants]
           if fam.reverse_template is not None else None)
    return {
        'label': fam.label,
        'top': [e.label for e in g.top],
        'tree': tree,
        'forward_template': fwd,
        'reverse_template': rev,
        'own_reverse': bool(fam.own_reverse),
    }


def _index_map(mol):
    """{atom: index} for the vertices of an RMG Molecule/Group, in vertex order."""
    return {a: i for i, a in enumerate(mol.vertices)}


def _normalize_mapping(mapping, react_idx, tmpl_idx):
    """A mapping is {reactant_atom: template_atom}. Normalize to a sorted list
    of (reactant_atom_index, template_atom_index). Both indices are the stable
    vertex-list positions within their respective (unsorted) graphs, so the
    representation is identical across the rmgpy and rmgpu implementations.
    (Template recipe labels are NOT used here: the template atoms in
    _match_reactant_to_template are the raw template-reactant group atoms, and
    their labels can be empty; the vertex index is the stable identity.)"""
    pairs = []
    for react_atom, tmpl_atom in mapping.items():
        pairs.append((react_idx[react_atom], tmpl_idx[tmpl_atom]))
    return sorted(pairs)


def _resolve_groups(item, entries):
    """Resolve a template-reactant item to its concrete Group components.
    A Group is itself; a LogicNode expands to its possible structures (RMG
    _match_reactant_to_template iterates get_possible_structures, passing the
    family's groups.entries to resolve the component labels)."""
    from rmgpy.data.base import LogicNode
    if isinstance(item, LogicNode):
        return [g for g in item.get_possible_structures(entries)
                if hasattr(g, 'vertices')]
    if hasattr(item, 'vertices'):
        return [item]
    return []


def _match_case(fam, smiles):
    """Record RMG's own matching results for the case (base reactants)."""
    reactants = [Molecule().from_smiles(s) for s in smiles]
    template_reactants = [x.item for x in fam.forward_template.reactants]
    entries = fam.groups.entries
    # single-group split (e.g. R_Recombination Y_rad -> two reactant slots)
    n_tr = len(template_reactants)
    if len(reactants) > n_tr and n_tr == 1:
        grp = template_reactants[0]
        if hasattr(grp, 'split'):
            try:
                template_reactants = list(grp.split())
            except AttributeError:
                pass
    # Build, per slot, the concrete groups and a GLOBAL template-atom index
    # (offset per component group) so mappings normalize stably.
    slot_groups = []      # list of lists of Group
    slot_tidx = []        # {template_atom: global_index}
    for tr in template_reactants:
        grps = _resolve_groups(tr, entries)
        tidx = {}
        off = 0
        for g in grps:
            for a in g.vertices:
                tidx[a] = off
                off += 1
        slot_groups.append(grps)
        slot_tidx.append(tidx)
    # Concrete per-slot group adjlists (post-split, LogicOr-expanded) so the
    # rmgpu test can parse + match without re-deriving the family structure.
    slot_groups_adj = [[g.to_adjacency_list() for g in grps]
                       for grps in slot_groups]
    react_idx = [_index_map(r) for r in reactants]
    per_slot = []
    for ri, r in enumerate(reactants):
        for ti, tr in enumerate(template_reactants):
            try:
                maps = fam._match_reactant_to_template(r, tr)
            except Exception as e:
                maps = []
                _err = '%s: %s' % (type(e).__name__, e)
            else:
                _err = None
            per_slot.append({
                'reactant_index': ri,
                'reactant_smiles': smiles[ri],
                'template_slot': ti,
                'template_label': getattr(tr, 'label', None),
                'n_components': len(slot_groups[ti]),
                'n_mappings': len(maps),
                'mappings': [
                    sorted((react_idx[ri][ra], slot_tidx[ti][ta])
                           for ra, ta in m.items())
                    for m in maps
                ],
                'error': _err,
            })
    # Most-specific template labels via RMG's tree walk (needs a Reaction with
    # labeled reactants; label them with RMG's own forward-template matching).
    labels = {'forward_template': [e.label for e in fam.forward_template.reactants],
              'most_specific': None, 'error': None}
    try:
        # add_atom_labels_for_reaction mutates the Reaction; we only need the
        # reactant labels to then run get_reaction_template.
        rxn = Reaction(reactants=reactants,
                       products=[Molecule().from_smiles(smiles[0])])
        labeled = fam.get_labeled_reactants_and_products(
            reactants, [Molecule().from_smiles(smiles[0])])
        if labeled[0] is not None:
            labeled_reactants = labeled[0]
            rxn2 = Reaction(reactants=labeled_reactants,
                            products=labeled[1] if labeled[1] else rxn.products)
            labels['most_specific'] = fam.get_reaction_template_labels(rxn2)
        else:
            labels['error'] = 'get_labeled_reactants_and_products returned None'
    except Exception as e:
        labels['error'] = '%s: %s' % (type(e).__name__, e)
    return {'family': fam.label, 'reactant_smiles': smiles,
            'per_slot': per_slot, 'labels': labels,
            'slot_groups_adj': slot_groups_adj,
            'n_template_slots': len(template_reactants)}


def _construct_inventory(family_records):
    inv = {k: [] for k in CONSTRUCT_RE}
    for fam, rec in family_records.items():
        for label, node in rec['tree'].items():
            if node['node']['type'] != 'group':
                continue
            adj = node['node']['adj']
            for key, rx in CONSTRUCT_RE.items():
                if rx.search(adj):
                    if label not in inv[key]:
                        inv[key].append('%s::%s' % (fam, label))
    return {k: len(v) for k, v in inv.items()}, inv


def main():
    out = {'atomtype_tree': _atomtype_tree()}

    real_db = KineticsDatabase()
    real_db.load_families(
        path=settings['database.directory'] + '/kinetics/families',
        families=FAMILIES['real'])
    test_db = KineticsDatabase()
    test_db.load_families(
        path=settings['test_data.directory'] + '/testing_database/kinetics/families',
        families=FAMILIES['test'])

    family_records = {}
    fam_obj = {}
    for db in (real_db, test_db):
        for f in db.families:
            fam = db.families[f]
            fam_obj[f] = fam
            family_records[f] = _record_family(fam)
    out['families'] = family_records
    _counts, _detail = _construct_inventory(family_records)
    out['construct_inventory'] = _counts

    # Round-trip matching ground truth per case.
    cases_out = []
    for fam_label, source, smiles in CASES:
        fam = fam_obj[fam_label]
        rec = _match_case(fam, smiles)
        rec['source'] = source
        cases_out.append(rec)
    out['match_cases'] = cases_out

    out_path = os.path.join(REPO, 'gates', 'baselines', 'job05',
                            'step03_templates_reference.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', out_path)
    print('families:', sorted(family_records))
    print('atomtypes:', len(out['atomtype_tree']))
    print('match_cases:', len(cases_out))
    for c in cases_out:
        n = sum(s['n_mappings'] for s in c['per_slot'])
        print('  %-26s %-22s mappings=%d most_specific=%s%s' % (
            c['family'], ','.join(c['reactant_smiles']), n,
            c['labels']['most_specific'],
            '  ERR:' + c['labels']['error'] if c['labels']['error'] else ''))


if __name__ == '__main__':
    main()
