"""Probe 6: current per-form Intra_ene attribution (which forms -> which products)."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu/gates')
from rmgpu.molecule.molecule import Molecule
from rmgpu.core.family import KineticsFamilies
from rmgpu.core.template import TemplateMatcher
from rmgpu.core import enumeration as enum

kf = KineticsFamilies().load('default')
fam = {f.label: f for f in kf.families}['Intra_ene_reaction']
efam = enum.Family(
    label=fam.label, recipe=fam.recipe, reverse_recipe=fam.reverse_recipe,
    own_reverse=fam.own_reverse, reversible=fam.reversible,
    allow_charged_species=fam.allow_charged_species, electrons=fam.electrons,
    reactant_num_effective=fam.num_template_reactants_effective,
    product_num_forward=fam.product_num_forward,
    reverse_map=fam.reverse_map,
    template_labels=[e.label for e in fam.forward_template],
    forbidden=fam.forbidden,
)
matcher = TemplateMatcher(fam)
efam.matcher = matcher

smi = 'C[CH]C1=CC=CC=C1'
mol = Molecule(smiles=smi)

# replicate _enumerate_fresh's setup
species = enum.expand_resonance([mol])
enum.assign_fresh_ids(species)
print('forms after assign_fresh_ids: %d' % len(species[0]))

def canon(m):
    from rdkit import Chem
    r = m._rdkit
    if not any(a.GetSymbol() == 'H' for a in r.GetAtoms()):
        try: r = Chem.AddHs(r)
        except Exception: pass
    try: Chem.Kekulize(r, clearAromaticFlags=True)
    except Exception: pass
    try: Chem.SanitizeMol(r)
    except Exception: pass
    return Chem.MolToSmiles(r)

for i, form in enumerate(species[0]):
    mappings = matcher.match_molecule(form, 0, 'ab')
    print('form[%d] %s  matchings=%d' % (i, form.to_smiles(), len(mappings)))
    prods = set()
    for mapping in mappings:
        enum._set_labels(form, mapping)
        rxn = enum._apply_one_application(efam, [form], True)
        if rxn is not None:
            for p in rxn.products:
                prods.add(canon(p))
        enum.clear_labeled_atoms([form])
    for p in sorted(prods):
        print('    -> product %s' % p)

print('\n=== full generate_reactions (collapsed) ===')
rxns = enum.generate_reactions(efam, [Molecule(smiles=smi)], matcher=matcher)
for r in rxns:
    print('  deg=%.1f products=%s' % (r.degeneracy, [canon(p) for p in r.products]))
