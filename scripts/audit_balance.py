"""Fast atom-balance audit of every reaction in the live superminimal it-3 sim
(no integration). Uses RDKit formulas (not a crude regex). Flags any reaction
that creates/destroys atoms.
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 /home/jackson/miniforge3/envs/rmgpu/bin/python scripts/audit_balance.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.disable(logging.CRITICAL)

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from probe_live_sm import build_loop

import re
from collections import Counter


def atom_counts(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    Chem.SanitizeMol(m)
    if not any(a.GetSymbol() == "H" for a in m.GetAtoms()):
        m = Chem.AddHs(m)
    cnt = Counter(a.GetSymbol() for a in m.GetAtoms())
    return dict(cnt)


def main():
    loop, ctx, T, P = build_loop()
    for it in range(1, 4):
        loop.model.iteration_num = it
        loop.enlarge(it)
        loop.simulate(it)
        # don't promote/demote/prune - keep the full it-3 reaction set
    sim = loop._last_sim
    keys, nu, rps = sim["keys"], sim["nu"], sim["rps"]
    print(f"n_sp={len(keys)} n_rx={len(rps)}", flush=True)
    bad = 0
    for j in range(len(rps)):
        ra = Counter()
        pa = Counter()
        for i, c in enumerate(nu[j]):
            cnt = atom_counts(keys[i])
            if cnt is None:
                continue
            if c < 0:
                for el, n in cnt.items():
                    ra[el] += n * (-c)
            elif c > 0:
                for el, n in cnt.items():
                    pa[el] += n * c
        rkey = "+".join(keys[i] for i, c in enumerate(nu[j]) if c < 0)
        pkey = "+".join(keys[i] for i, c in enumerate(nu[j]) if c > 0)
        if ra != pa:
            bad += 1
            print(f"  UNBALANCED rxn{j:2d} ({rps[j].family:20s}) "
                  f"{rkey:26s}>>{pkey:22s} react={dict(ra)} prod={dict(pa)}",
                  flush=True)
        else:
            print(f"  ok         rxn{j:2d} ({rps[j].family:20s}) "
                  f"{rkey:26s}>>{pkey:22s}", flush=True)
    print(f"\nTOTAL UNBALANCED: {bad}/{len(rps)}", flush=True)


if __name__ == "__main__":
    main()
