"""Diagnose the superminimal NaN: dump the per-reaction rate constants at T/P
and find which reaction has a non-finite or absurd A(T)/reverse factor, and
show the profile around the blow-up point (last finite row vs next).
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 /home/jackson/miniforge3/envs/rmgpu/bin/python scripts/diag_nan.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchdae  # noqa: E402
import logging  # noqa: E402
logging.disable(logging.CRITICAL)

from probe_solvers import build_loop, rate_system  # noqa: E402
from rmgpu.reactor.simulator import (characteristic_rate, forward_A_T,  # noqa: E402
                                     reverse_factor, R)

EPS = 1e-30


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
    char = characteristic_rate(sim["nu"], sim["rps"], T, P, sim["init"])
    t_end = min(5.0 / char, float(ctx.termination_time or 1e12))
    keys, nu, rps = sim["keys"], sim["nu"], sim["rps"]
    c_tot = P / (R * T)
    print(f"n_sp={len(keys)} n_rx={len(rps)} T={T} P={P} t_end={t_end:.6g}")
    print("\n=== per-reaction rate constants ===")
    for j, rp in enumerate(rps):
        n_react = sum(-m for m in nu[j] if m < 0)
        n_prod = sum(m for m in nu[j] if m > 0)
        aT = forward_A_T(rp, T)
        rt = reverse_factor(rp, T, float(n_prod - n_react))
        kf = aT * c_tot ** (n_react - 1)
        kr = aT * rt * c_tot ** (n_prod - 1)
        flag = ""
        if not (np.isfinite(kf) and np.isfinite(kr)):
            flag = "  <-- NONFINITE"
        elif abs(np.log10(max(kf, kr))) > 40:
            flag = "  <-- ABSURD (>1e40)"
        print(f"rxn {j:2d} A={rp.A:.3e} n={rp.n:6.2f} Ea={rp.Ea:10.1f} "
              f"dS={rp.dS:9.2f} dH={rp.dH:10.1f} fam={rp.family} "
              f"src={rp.source} A_T={aT:.3e} rev={rt:.3e} kf={kf:.3e} kr={kr:.3e}{flag}")
        if flag:
            print(f"     nu={['%+.0f' % m for m in nu[j]]} "
                  f"react={keys[[i for i, m in enumerate(nu[j]) if m < 0]]} "
                  f"prod={keys[[i for i, m in enumerate(nu[j]) if m > 0]]}")

    # integrate with n=256 and dump around the blow-up
    dydt, device = rate_system(keys, nu, rps, T, P)
    y0_t = torch.tensor(sim["init"], dtype=torch.float64, device=device)[None, :]
    yp0 = dydt(y0_t)

    def F(t, y, yp):
        y1 = y if y.dim() == 1 else y[0]
        return yp - dydt(y1[None, :])[0]

    h = max(t_end / 256, 1e-18)
    sol = torchdae.solve_tr_bdf2(F, (0.0, float(t_end)), y0_t, h=float(h), yp0=yp0)
    ts = sol.ts.cpu().numpy()
    ys = sol.ys.squeeze(1).cpu().numpy()
    fin = np.all(np.isfinite(ys), axis=1)
    last_fin = np.where(fin)[0][-1]
    print(f"\n=== profile around blow-up (n=256): last finite row idx {last_fin} ===")
    for i in range(max(0, last_fin - 3), min(len(ts), last_fin + 4)):
        row = ys[i]
        s = row.sum()
        top = np.argsort(-np.abs(np.nan_to_num(row)))[:4]
        print(f"t={ts[i]:12.4g} sum={s:10.5f} finite={bool(fin[i])} "
              f"top={[ (keys[k], f'{row[k]:.3e}') for k in top ]}")


if __name__ == "__main__":
    main()
