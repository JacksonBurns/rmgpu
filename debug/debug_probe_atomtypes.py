#!/usr/bin/env python3
"""Dump RMG-Py atom type feature requirements for the types seen in the 19 test molecules.
Reads the live atomtype DB (the authoritative source) so rmgpu can replicate the table."""
import sys
sys.path.insert(0, '/home/jackson/rmgpu/RMG-Py')
from rmgpy.molecule.atomtype import AtomType

# Types observed in the probe:
wanted = ['H0','Cs','Cd','CO','Cdd','Ct','Cb','Cbf','Cs2',
          'N3s','N3d','N3t','N1s','N5sc','N5dc','N1d','N2s','N3sc','N3dc',
          'O2s','O2d','O4sc','O4dc','O0sc','O1s','O3s','O2sc','O2dc',
          'S2s','S2d','S4s']
seen = set()
for t in wanted:
    try:
        at = AtomType(t)
    except Exception as e:
        print(f"MISSING {t}: {e}")
        continue
    # dump every feature attribute
    feats = {}
    for attr in ['label','symbol','radical_electrons','charge','lone_pairs',
                 'single','double','triple','quadruple','bond','aromatic','benzene',
                 'r_double','o_double','s_double','r_aromatic','o_aromatic',
                 's_aromatic','r_single','o_single','s_single','r_triple','o_triple','s_triple',
                 'r_quadruple','o_quadruple','s_quadruple']:
        if hasattr(at, attr):
            feats[attr] = getattr(at, attr)
    print(f"TYPE {t}:")
    for k, v in feats.items():
        if v not in (0, [], None, False):
            print(f"    {k} = {v!r}")
