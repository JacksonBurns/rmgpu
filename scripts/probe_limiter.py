"""Validate a maximum-rate limiter on the live superminimal it-3 sim (the
mechanism that overflows to NaN with plain TR-BDF2 n=32). Test:
  (a) plain TR-BDF2 n=32           (current code)  -> expect NaN
  (b) TR-BDF2 n=32 + rate limiter  (CFL: |net_j| <= RMAX)
  (c) TR-BDF2 n=32 + per-step y overshoot guard
Check whether (b)/(c) yield a physical final profile (finite, in [0,1], sum 1).
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 /home/jackson/miniforge3/envs/rmgpu/bin/python scripts/probe_limiter.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchdae  # noqa: E402
import logging  # noqa: E402
logging.disable(logging.CRITICAL)

from probe_live_sm import build_loop, rate_system  # noqa: E402
from rmgpu.reactor.simulator import characteristic_rate  # noqa: E402


def build_limited(nu, rps, T, P, h, rmax):
    """Same ODE as rate_system, but cap each reaction net rate to |rmax|."""
    dydt_base, device, _ = rate_system(nu, rps, T, P)
    # Rebuild with cap: simplest is to wrap dydt and clip the net-rate vector.
    # We rebuild directly (mirror of rate_system) so we can clamp net.
    from rmgpu.reactor.simulator import (forward_A_T, reverse_factor, R)
    n_rx = len(rps)
    n_react_list, n_prod_list, dnu_list = [], [], []
    react_i, react_c, prod_i, prod_c, nu_i, nu_c = [], [], [], [], [], []
    for j in range(n_rx):
        ri, rc, pi, pc, ni, nc = [], [], [], [], [], []
        n_r = n_p = 0
        dnu = 0.0
        for i, c in enumerate(nu[j]):
            if c < 0:
                ri.append(i); rc.append(-c); n_r += -c; dnu += c
                ni.append(i); nc.append(c)
            elif c > 0:
                pi.append(i); pc.append(c); n_p += c; dnu += c
                ni.append(i); nc.append(c)
        react_i.append(torch.tensor(ri, dtype=torch.long, device=device))
        react_c.append(torch.tensor(rc, dtype=torch.float64, device=device))
        prod_i.append(torch.tensor(pi, dtype=torch.long, device=device))
        prod_c.append(torch.tensor(pc, dtype=torch.float64, device=device))
        nu_i.append(torch.tensor(ni, dtype=torch.long, device=device))
        nu_c.append(torch.tensor(nc, dtype=torch.float64, device=device))
        n_react_list.append(n_r)
        n_prod_list.append(n_p)
        dnu_list.append(dnu)
    n_react_t = torch.tensor(n_react_list, dtype=torch.float64, device=device)
    n_prod_t = torch.tensor(n_prod_list, dtype=torch.float64, device=device)
    dnu_t = torch.tensor(dnu_list, dtype=torch.float64, device=device)
    c_tot = P / (R * T)
    A_T = torch.tensor([forward_A_T(rp, T) for rp in rps], dtype=torch.float64, device=device)
    rev_t = torch.tensor([reverse_factor(rps[j], T, float(dnu_list[j])) for j in range(n_rx)],
                         dtype=torch.float64, device=device)
    k_fwd = A_T * torch.pow(torch.tensor(c_tot, dtype=torch.float64, device=device), n_react_t - 1.0)
    k_rev = A_T * rev_t * torch.pow(torch.tensor(c_tot, dtype=torch.float64, device=device), n_prod_t - 1.0)
    eps = 1e-30

    def dydt(y):
        batch = y.shape[0]
        net = []
        for j in range(n_rx):
            if len(react_i[j]) > 0:
                y_r = torch.clamp(y[:, react_i[j]], min=eps)
                f = torch.ones(batch, device=device)
                for k in range(len(react_i[j])):
                    f = f * (y_r[:, k] ** react_c[j][k])
            else:
                f = torch.ones(batch, device=device)
            if len(prod_i[j]) > 0:
                y_p = torch.clamp(y[:, prod_i[j]], min=eps)
                rf = torch.ones(batch, device=device)
                for k in range(len(prod_i[j])):
                    rf = rf * (y_p[:, k] ** prod_c[j][k])
            else:
                rf = torch.ones(batch, device=device)
            netj = k_fwd[j] * f - k_rev[j] * rf
            netj = torch.clamp(netj, -rmax, rmax)  # <-- limiter
            net.append(netj)
        net = torch.stack(net, dim=1)
        out = torch.zeros_like(y)
        for j in range(n_rx):
            nj = net[:, j]
            for k in range(len(nu_i[j])):
                out[:, nu_i[j][k]] += nj * nu_c[j][k]
        dnu_sum = (net * dnu_t.unsqueeze(0)).sum(dim=1, keepdim=True)
        return out - y * dnu_sum

    return dydt, device


def evaluate(name, ts, ys, keys):
    fin = np.all(np.isfinite(ys), axis=1)
    n_fin = int(fin.sum())
    if fin[-1]:
        last = ys[-1]
        mn, mx, s = float(last.min()), float(last.max()), float(last.sum())
        phys = (mn >= -1e-6) and (mx <= 1 + 1e-6) and abs(s - 1) < 1e-3
        top = [(keys[k], round(float(last[k]), 5)) for k in np.argsort(-last)[:5]]
        print(f"{name:26s} {n_fin}/{len(ts)} finite  PHYSICAL={phys} "
              f"min={mn:.2e} max={mx:.2e} sum={s:.5f} top={top}", flush=True)
    else:
        fb = np.where(~fin)[0][0]
        print(f"{name:26s} {n_fin}/{len(ts)} finite  first-bad row {fb} t={ts[fb]:.4g}",
              flush=True)


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
    h = max(t_end / 32, 1e-18)
    print(f"n_sp={len(keys)} n_rx={len(rps)} t_end={t_end:.6g} h={h:.4g}", flush=True)

    dydt, device = rate_system(nu, rps, T, P)
    y0_t = torch.tensor(init, dtype=torch.float64, device=device)[None, :]
    yp0 = dydt(y0_t)

    def F_plain(t, y, yp):
        y1 = y if y.dim() == 1 else y[0]
        return yp - dydt(y1[None, :])[0]

    sol = torchdae.solve_tr_bdf2(F_plain, (0.0, float(t_end)), y0_t, h=float(h), yp0=yp0)
    evaluate("(a) plain tr_bdf2 n=32", sol.ts.cpu().numpy(),
             sol.ys.squeeze(1).cpu().numpy(), keys)

    # characteristic max rate at init (to scale the limiter)
    for rmax in (1.0e6, 1.0e10, 1.0e14):
        dydt_l, dev = build_limited(nu, rps, T, P, h, rmax)
        y0_l = torch.tensor(init, dtype=torch.float64, device=dev)[None, :]
        yp0_l = dydt_l(y0_l)

        def F_l(t, y, yp, _dy=dydt_l):
            y1 = y if y.dim() == 1 else y[0]
            return yp - _dy(y1[None, :])[0]
        try:
            sol = torchdae.solve_tr_bdf2(F_l, (0.0, float(t_end)), y0_l, h=float(h), yp0=yp0_l)
            evaluate(f"(b) limited rmax={rmax:.0e}", sol.ts.cpu().numpy(),
                     sol.ys.squeeze(1).cpu().numpy(), keys)
        except Exception as e:
            print(f"(b) limited rmax={rmax:.0e}: EXC {type(e).__name__}: {str(e)[:100]}",
                  flush=True)


if __name__ == "__main__":
    main()
