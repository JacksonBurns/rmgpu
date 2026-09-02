"""RMG-Py ground truth for the Intra_ene benzylic-radical case."""
import sys
sys.path.insert(0, '/home/jackson/rmg/RMG-Py')
from rmgpy.molecule.molecule import Molecule
from rmgpy.data.kinetics.database import KineticsDatabase
from rmgpy.data.kinetics.family import KineticsFamily

DBDIR = '/home/jackson/rmg/RMG-database/input'

# Build the benzylic radical (aromatic input)
m = Molecule()
m.from_smiles('C[CH]c1ccccc1')
print('input smiles:', m.to_smiles())
print('is_aromatic:', m.is_aromatic())
m.update()

# Resonance forms
forms = m.generate_resonance_structures(keep_isomorphic=True)
print('\nRMG resonance forms: %d' % len(forms))
for i, f in enumerate(forms):
    f.update()
    print('  form[%d] %s  reactive=%s' % (i, f.to_smiles(), f.reactive))

# Load the Intra_ene family and get its template group
db = KineticsDatabase(directory=DBDIR)
fam = db.loadFamily('Intra_ene_reaction')
print('\nfamily: %s' % fam.label)
# The template reactant group
tmpl = fam.templateReactants
print('templateReactants:', len(tmpl))
tgroup = tmpl[0]
print('template reactant group atom count:', len(tgroup.atoms))

# Per-form matching: for each resonance form, find subgraph isomorphisms of the template
print('\n=== per-form matching counts (template -> form) ===')
for i, f in enumerate(forms):
    f.update()
    # RMG: group.match_to_structure or find_subgraph_isomorphisms
    try:
        mappings = tgroup.find_subgraph_isomorphisms(f)
        print('  form[%d] %s  -> %d matchings (reactive=%s)' % (
            i, f.to_smiles(), len(mappings), f.reactive))
    except Exception as e:
        print('  form[%d] %s  -> ERROR %s' % (i, f.to_smiles(), e))

# Now run the family's generate_reactions on the benzylic radical
print('\n=== family.generate_reactions on the benzylic radical ===')
reactants = [m]
rxns = fam.generate_reactions(reactants)
print('num reactions:', len(rxns))
for r in rxns:
    prods = r.products
    print('  deg=%.1f  family=%s  products=' % (r.degeneracy, r.family.label),
          [p.to_smiles() for p in prods])
