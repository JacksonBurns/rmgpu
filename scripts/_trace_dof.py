import sys
sys.path.insert(0,'.')
from rdkit import Chem
from rmgpu.molecule.molecule import Molecule
from rmgpu.core import enumeration as enum
import rmgpu.core.recipe as R
import json
ref = json.load(open('gates/baselines/job05/step02_products_reference.json'))
case = None
for c in ref['cases']:
    if c['family']=='R_Addition_MultipleBond' and c['reactant_smiles']==['C1=CC=CC=C1','[H]']:
        case = c; break
fam = enum.Family.from_reference(case)
reactants = [Molecule(smiles=s) for s in case['reactant_smiles']]
matcher = enum.RecordedMatcher(case)
apps = list(matcher.yield_applications())
structures = apps[0]
merged = R._merge_molecules(structures)
R._re_aromatize(merged)
R._kekulize_piece = lambda m: None  # skip kekulize to inspect raw state
fam.recipe._apply(merged, True, True)
m = Chem.RWMol(merged)
print("fractional bonds:", [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), R._bond_order(b)) for b in m.GetBonds() if R._is_fractional(R._bond_order(b))])
rings = R._all_six_rings(m)
print("rings:", rings)
ring = rings[0]; rs=set(ring)
endo=set(); exo=set(); aromatic=True
for i in range(6):
    a1=ring[i]; atom1=m.GetAtomWithIdx(a1)
    nbrs_in=[n.GetIdx() for n in atom1.GetNeighbors() if n.GetIdx() in rs]
    bridged=len(nbrs_in)>2
    for nbr in atom1.GetNeighbors():
        a2=nbr.GetIdx(); bond=m.GetBondBetweenAtoms(a1,a2)
        if bridged and len([n.GetIdx() for n in nbr.GetNeighbors() if n.GetIdx() in rs])>2:
            exo.add(frozenset((a1,a2))); continue
        elif a2 in rs:
            o=R._bond_order(bond)
            if abs(round(o)-o)<1e-9:
                aromatic=False; break
            endo.add(frozenset((a1,a2)))
        else:
            exo.add(frozenset((a1,a2)))
    if not aromatic: break
print("aromatic:",aromatic,"endo:",[sorted(k) for k in endo],"exo:",[sorted(k) for k in exo])
ar=R._AromaticRing(m, ring, endo, exo)
ar.update()
print("endo_dof",ar.endo_dof,"exo_dof",ar.exo_dof)
print("unresolved:",[sorted(b.key) for b in ar.unresolved])
# instrument: run the kekulize loop by hand with tracing
def trace():
    it=0; maxit=2*len(ar.unresolved)
    while ar.unresolved and it<maxit:
        for b in ar.unresolved:
            b.update()
        ar.unresolved.sort(key=lambda b:(b.double_possible,not b.double_required,b.endo_dof,b.exo_dof),reverse=True)
        bond=ar.unresolved.pop()
        tag="S/D req" if (bond.double_possible and bond.double_required) else ("possible" if bond.double_possible else "impossible")
        print("  it=%d pop %s %s poss=%s req=%s endo_dof=%d exo_dof=%d" % (it,sorted(bond.key),tag,bond.double_possible,bond.double_required,bond.endo_dof,bond.exo_dof))
        if bond.double_possible and bond.double_required:
            bond.order=2; ar.resolved.append(bond); ar.endo_dof-=1
        elif bond.double_possible and not bond.double_required:
            if ((ar.endo_dof==6 and ar.exo_dof==0) or (ar.endo_dof==6 and bond.exo_dof==0) or (bond.endo_dof==1 and (bond.exo_dof==1 or bond.exo_dof==2)) or ar.endo_dof==1):
                bond.order=2; ar.resolved.append(bond); ar.endo_dof-=1
                print("    -> assumed double")
            else:
                ar.unresolved.append(bond); print("    -> deferred")
        else:
            bond.order=1; ar.resolved.append(bond); ar.endo_dof-=1
            print("    -> single")
        it+=1
    print("done, unresolved:",[sorted(b.key) for b in ar.unresolved])
trace()
print("SMILES:", Chem.MolToSmiles(m, canonical=True))
