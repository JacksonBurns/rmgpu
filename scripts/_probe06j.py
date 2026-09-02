"""Probe 10: lock in the form classifier (aromatic_rep / reactive)."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')
from rmgpu.molecule.molecule import Molecule
from rdkit import Chem

def ring_info(em):
    """For the largest 6-ring: bond orders, isAromatic flags, S/D pattern."""
    ri = em.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) == 6:
            orders = []
            arflags = []
            for k in range(6):
                b = em.GetBondBetweenAtoms(ring[k], ring[(k+1)%6])
                orders.append(round(b.GetBondTypeAsDouble(), 2))
                arflags.append(b.GetIsAromatic())
            pat = ''.join('D' if abs(o-2)<.1 else 'S' if abs(o-1)<.1 else '?' for o in orders)
            return pat, arflags
    return None, None

def re_arom(mol):
    try:
        em = Chem.Mol(mol._rdkit)
        Chem.SanitizeMol(em)
        Chem.SetAromaticity(em)
        return sum(1 for b in em.GetBonds() if b.GetIsAromatic())
    except Exception as e:
        return 'ERR %s' % e

def classify(mol):
    """Proposed classifier for a raw resonance form."""
    em = mol._with_explicit_h()
    pat, arflags = ring_info(em)
    # aromatic_rep: the 6-ring bonds are aromatic (isAromatic True)
    aromatic_rep = bool(arflags) and all(arflags)
    # standard kekulized benzene ring: S/D alternating, NOT aromatic
    std_kek = pat in ('SDSDSD', 'DSDSDS') and (arflags is None or not all(arflags))
    return aromatic_rep, std_kek, pat, arflags, re_arom(mol)

print('=== raw resonance forms of C[CH]C1=CC=CC=C1 (current gen) ===')
mol = Molecule(smiles='C[CH]C1=CC=CC=C1')
from rmgpu.molecule.resonance import generate_resonance_structures
forms = generate_resonance_structures(mol)
for i, f in enumerate(forms):
    ar, sk, pat, af, ra = classify(f)
    print('  [%d] %-22s aromatic_rep=%-5s std_kek=%-5s ring=%s arflags=%s re_arom=%s' % (
        i, f.to_smiles(), ar, sk, pat, af, ra))

print('\n=== plain benzene c1ccccc1 (for R_Addition / H_Abstraction cases) ===')
b = Molecule(smiles='c1ccccc1')
forms_b = generate_resonance_structures(b)
for i, f in enumerate(forms_b):
    ar, sk, pat, af, ra = classify(f)
    print('  [%d] %-22s aromatic_rep=%-5s std_kek=%-5s ring=%s arflags=%s re_arom=%s' % (
        i, f.to_smiles(), ar, sk, pat, af, ra))

print('\n=== kekulized benzene C1=CC=CC=C1 (the input form for R_Addition) ===')
bk = Molecule(smiles='C1=CC=CC=C1')
forms_bk = generate_resonance_structures(bk)
for i, f in enumerate(forms_bk):
    ar, sk, pat, af, ra = classify(f)
    print('  [%d] %-22s aromatic_rep=%-5s std_kek=%-5s ring=%s arflags=%s re_arom=%s' % (
        i, f.to_smiles(), ar, sk, pat, af, ra))
