from rdkit import Chem
got = '[H][c]1c([H])c([H])c([H])c([H])c-1([H])[H]'
want = '[H][C]1C([H])=C([H])C([H])=C([H])C1([H])[H]'
g = Chem.MolFromSmiles(got); w = Chem.MolFromSmiles(want)
from rdkit.Chem import rdMolDescriptors
print("got  formula:", rdMolDescriptors.CalcMolFormula(g), "radicals:", [a.GetNumRadicalElectrons() for a in g.GetAtoms()])
print("want formula:", rdMolDescriptors.CalcMolFormula(w), "radicals:", [a.GetNumRadicalElectrons() for a in w.GetAtoms()])
# isomorphism (explicit H)
ge = Chem.AddHs(Chem.Mol(g)); we = Chem.AddHs(Chem.Mol(w))
print("isomorphic (AddHs):", ge.IsSubstructMatch(we) and we.IsSubstructMatch(ge))
# try kekulizing the aromatic form
g2 = Chem.Mol(g)
try:
    Chem.Kekulize(g2, clearAromaticFlags=True)
    print("kekulized got ->", Chem.MolToSmiles(g2, canonical=True))
except Exception as e:
    print("kekulize got FAILED:", type(e).__name__, e)
# Does the aromatic form have aromatic flags?
print("got aromatic atoms:", [a.GetIsAromatic() for a in g.GetAtoms()])
print("want aromatic atoms:", [a.GetIsAromatic() for a in w.GetAtoms()])
