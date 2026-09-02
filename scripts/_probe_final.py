"""Probe final: Intra_ene benzylic case through the REAL gate path.
Expected (RMG-Py): A = C=CC1=CCC=C[CH]1 deg 6.0; B = C=CC1=CC[CH]C=C1 deg 3.0.
No spurious allenes.
"""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.molecule.resonance import generate_resonance_structures
from rmgpu.core import enumeration as enum
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
efam.matcher = matcher
mol = Molecule(smiles='C[CH]C1=CC=CC=C1')

def canon(p):
    r2 = p._rdkit
    if not any(a.GetSymbol() == 'H' for a in r2.GetAtoms()):
        try: r2 = Chem.AddHs(r2)
        except Exception: pass
    try: Chem.Kekulize(r2, clearAromaticFlags=True)
    except Exception: pass
    return Chem.MolToSmiles(r2)

rxns = enum.generate_reactions(efam, [mol], matcher=matcher)
print('Intra_ene C[CH]C1=CC=CC=C1 -> %d reactions' % len(rxns))
for r in rxns:
    print('  deg=%.1f products=%s' % (r.degeneracy, [canon(p) for p in r.products]))
print()
print('EXPECTED: A=C=CC1=CCC=C[CH]1 deg 6.0 ; B=C=CC1=CC[CH]C=C1 deg 3.0 ; NO allenes')
