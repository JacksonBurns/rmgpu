"""Find the minimum fixed step count (n = simulate_steps, h = t_end/n) that
keeps the superminimal final-mechanism profile finite over the full t_end.
Also report wall time per config to bound c3h4 feasibility.
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 /home/jackson/miniforge3/envs/rmgpu/bin/python scripts/probe_stepcount.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchdae  # noqa: E402
import logging  # noqa: E402
logging.disable(logging.CRITICAL)

from probe_solvers import build_loop, rate_system  # noqa: E402
from rmgpu.reactor.simulator import characteristic_rate  # noqa: E402


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
    t_end = max(1e-9, min(t_end, 1e12))
    keys, nu, rps, y0 = sim["keys"], sim["nu"], sim["rps"], sim["init"]
    print(f"n_sp={len(keys)} n_rx={len(rps)} t_end={t_end:.6g} s")
    dydt, device = rate_system(keys, nu, rps, T, P)
    y0_t = torch.tensor(y0, dtype=torch.float64, device=device)[None, :]
    yp0 = dydt(y0_t)

    def F(t, y, yp):
        y1 = y if y.dim() == 1 else y[0]
        return yp - dydt(y1[None, :])[0]

    for n in (64, 256, 1024):
        h = max(t_end / n, 1e-18)
        t0 = time.time()
        try:
            sol = torchdae.solve_tr_bdf2(F, (0.0, float(t_end)), y0_t, h=float(h),
                                         yp0=yp0)
            ts = sol.ts.cpu().numpy()
            ys = sol.ys.squeeze(1).cpu().numpy()
            finite = bool(np.all(np.isfinite(ys)))
            sums = np.sum(ys, axis=1)
            sumerr = float(np.max(np.abs(sums - 1.0))) if finite else float("nan")
            mx = float(np.max(np.abs(ys))) if finite else float("nan")
            print(f"n_steps={n:5d} steps={len(ts):5d} wall={time.time()-t0:8.1f}s "
                  f"finite={finite} sum_err={sumerr:.3e} max|y|={mx:.3e}")
        except Exception as e:
            print(f"n_steps={n:5d} EXC {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
