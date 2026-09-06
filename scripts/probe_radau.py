"""Two checks:
  1) Atom balance of every reaction in the live superminimal it-3 sim
     (formula from RDKit; flag any reaction that creates/destroys atoms).
  2) Radau IIA5 at n=128 / 512 on the live sim: is the final state physical
     (finite, in [0,1], sum 1)?
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 /home/jackson/miniforge3/envs/rmgpu/bin/python scripts/probe_radau.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchdae  # noqa: E402
import logging  # noqa: E402
logging.disable(logging.CRITICAL)

import rmgpu.main as M  # noqa: E402
from rmgpu.core.loop import CoreEdgeLoop, RunContext, LoopConfig, canonical_key  # noqa: E402
from rmgpu.reactor.simulator import (characteristic_rate, R, _choose_device,  # noqa: E402
                                     forward_A_T, reverse_factor)
from probe_live_sm import build_loop, rate_system  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcNumAtoms  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def formula_of(mol):
    m = mol._rdkit if hasattr(mol, "_rdkit") else mol
    try:
        from rdkit.Chem import rdMolDescriptors
        return rdMolDescriptors.CalcMolFormula(m)
    except Exception:
        return "?"


def main():
    loop, ctx, T, P = build_loop()
    for it in range(1, 4):
        loop.model.iteration_num = it
        loop.enlarge(it)
        profiles, promote, demote = loop.simulate(it)
        for k in promote:
            loop._promote(k)
        for k in demote:
            loop._demote(k)
        loop.prune()
    sim = loop._last_sim
    keys, nu, rps, init = sim["keys"], sim["nu"], sim["rps"], sim["init"]
    char = characteristic_rate(nu, rps, T, P, init)
    t_end = min(5.0 / char, float(ctx.termination_time or 1e12))
    t_end = max(1e-9, min(t_end, 1e12))
    print(f"n_sp={len(keys)} n_rx={len(rps)} t_end={t_end:.6g} s", flush=True)

    # ---- 1) atom balance ------------------------------------------------
    from rdkit import Chem
    mols = {}
    for k in keys:
        try:
            m = Chem.MolFromSmiles(k)
            Chem.SanitizeMol(m)
            mols[k] = Chem.MolToSmiles(m)
        except Exception:
            mols[k] = None
    print("\n=== species formulas ===", flush=True)
    for k in keys:
        print(f"  {k:15s} -> {mols[k]}", flush=True)
    print("\n=== reaction atom balance ===", flush=True)
    for j, rp in enumerate(rps):
        rkey = "+".join(keys[i] for i, m in enumerate(nu[j]) if m < 0)
        pkey = "+".join(keys[i] for i, m in enumerate(nu[j]) if m > 0)
        # atom count via formula parsing
        def atoms(smi, mult=1):
            out = {}
            if smi is None:
                return out
            import re
            for el, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", smi.replace("[]", "")):
                if el and el[0].isupper():
                    out[el] = out.get(el, 0) + mult * int(cnt or 1)
            return out
        ra = {}
        pa = {}
        for i, m in enumerate(nu[j]):
            smi = mols[keys[i]]
            if m < 0:
                a = atoms(smi, -m)
                for e, c in a.items():
                    ra[e] = ra.get(e, 0) + c
            elif m > 0:
                a = atoms(smi, m)
                for e, c in a.items():
                    pa[e] = pa.get(e, 0) + c
        bal = "OK" if ra == pa else f"UNBALANCED react={ra} prod={pa}"
        print(f"  rxn{j:2d} {rkey:28s}>>{pkey:24s} {bal}", flush=True)

    # ---- 2) Radau at higher n -------------------------------------------
    dydt, device, _ = rate_system(nu, rps, T, P)
    y0_t = torch.tensor(init, dtype=torch.float64, device=device)[None, :]
    yp0 = dydt(y0_t)

    def F(t, y, yp):
        y1 = y if y.dim() == 1 else y[0]
        return yp - dydt(y1[None, :])[0]

    import time
    for n in (128, 512, 2048):
        h = max(t_end / n, 1e-18)
        t0 = time.time()
        try:
            sol = torchdae.solve_radau_iia5(F, (0.0, float(t_end)), y0_t, h=float(h),
                                            yp0=yp0)
            ts = sol.ts.cpu().numpy()
            ys = sol.ys.squeeze(1).cpu().numpy()
            fin = np.all(np.isfinite(ys), axis=1)
            n_fin = int(fin.sum())
            wall = time.time() - t0
            if fin[-1]:
                last = ys[-1]
                mn, mx = float(last.min()), float(last.max())
                s = float(last.sum())
                phys = (mn >= -1e-6) and (mx <= 1 + 1e-6) and abs(s - 1) < 1e-3
                top = [(keys[k], round(float(last[k]), 5)) for k in np.argsort(-last)[:5]]
                print(f"radau n={n:4d}: {n_fin}/{len(ts)} finite wall={wall:6.1f}s "
                      f"PHYSICAL={phys} min={mn:.2e} max={mx:.2e} sum={s:.5f} top={top}",
                      flush=True)
            else:
                fb = np.where(~fin)[0][0]
                print(f"radau n={n:4d}: {n_fin}/{len(ts)} finite wall={wall:6.1f}s "
                      f"first-bad row {fb} t={ts[fb]:.4g}", flush=True)
        except Exception as e:
            print(f"radau n={n:4d}: EXC {type(e).__name__}: {str(e)[:120]} "
                  f"(wall {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
