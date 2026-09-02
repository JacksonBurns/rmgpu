"""RMG-Py ground truth for Intra_ene benzylic radical (full flow)."""
import sys
sys.path.insert(0, '/home/jackson/rmg/RMG-Py')
from rmgpy import settings
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.molecule import Molecule

DBDIR = settings['database.directory']

# benzylic radical, aromatic input
m = Molecule()
m.from_smiles('C[CH]c1ccccc1')
print('input to_smiles:', m.to_smiles())
print('input is_aromatic:', m.is_aromatic())
m.update()

forms = m.generate_resonance_structures(keep_isomorphic=True)
print('\nRMG resonance forms: %d' % len(forms))
for i, f in enumerate(forms):
    f.update()
    # bond orders of the 6-ring
    orders = []
    try:
        rings = f.get_smallest_set_of_smallest_rings()
        for ring in rings:
            if len(ring) == 6:
                edges = f.get_edges_in_cycle(ring)
                orders = [b.get_order_str() for b in edges]
    except Exception as e:
        orders = ['ERR %s' % e]
    print('  form[%d] %s  reactive=%s  ring=%s' % (
        i, f.to_smiles(), f.reactive, ''.join(orders)))

# Load Intra_ene family
db = KineticsDatabase()
db.load_families(path=DBDIR + '/kinetics/families', families=['Intra_ene_reaction'])
fam = db.families['Intra_ene_reaction']
tgroup = fam.templateReactants[0]
print('\nIntra_ene template reactant group: %d atoms' % len(tgroup.atoms))

print('\n=== per-form subgraph matchings (template -> form) ===')
for i, f in enumerate(forms):
    f.update()
    try:
        mappings = tgroup.find_subgraph_isomorphisms(f)
        print('  form[%d] %s  -> %d matchings (reactive=%s)' % (
            i, f.to_smiles(), len(mappings), f.reactive))
    except Exception as e:
        print('  form[%d] -> ERROR %s: %s' % (i, type(e).__name__, e))

print('\n=== family.generate_reactions (the ground truth) ===')
from rmgpy.data.kinetics.common import (check_for_same_reactants,
    ensure_independent_atom_ids, find_degenerate_reactions,
    generate_molecule_combos)
reactants = [m.copy(deep=True)]
reactants, same = check_for_same_reactants(reactants)
ensure_independent_atom_ids(reactants, resonance=True)
raw = []
for combo in generate_molecule_combos(reactants):
    raw.extend(fam.generate_reactions(list(combo)))
fwd = [r for r in raw if r.is_forward]
fwd_c = find_degenerate_reactions(fwd, same_reactants=same, template=None,
                                  kinetics_family=fam, resonance=True)
print('num fwd collapsed reactions:', len(fwd_c))
for r in fwd_c:
    print('  deg=%.1f products=%s' % (r.degeneracy,
           [p.to_smiles() for p in r.products]))
