"""Adaptive TR-BDF2 (step-doubling local error control) built on
torchdae.bdf.tr_bdf2_step with strategy='always'.
Usage: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 /home/jackson/miniforge3/envs/rmgpu/bin/python scripts/probe_adaptive.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchdae
from torchdae.bdf import tr_bdf2_step
import logging
logging.disable(logging.CRITICAL)

from probe_live_sm import build_loop, rate_system
from rmgpu.reactor.simulator import characteristic_rate


def do_step(F, t, h, y, yp, step_tol=1e-8):
    return tr_bdf2_step(F, t, h, y, yp, tol=step_tol, damping=1.0,
                        strategy="always")


def adaptive_tr_bdf2(F, dydt, t0, t_end, y0, h_max, h_start=None,
                     rtol=1e-4, atol=1e-9, h_floor=1e-15, max_retries=60):
    device = y0.device
    y = y0.clone()
    yp = dydt(y)
    t = float(t0)
    ts = [t]
    ys = [y[0].clone().cpu()]
    h0 = h_max if h_start is None else min(h_start, h_max)
    h = min(h0, t_end - t)
    n_steps = 0
    n_retries = 0
    n_fallback = 0
    while t < t_end - 1e-12 * max(1.0, t_end):
        h_try = min(h, t_end - t)
        err = float("inf")
        yb = ypb = None
        attempt = 0
        for attempt in range(max_retries):
            y1, yp1 = do_step(F, t, h_try, y, yp)
            h2 = 0.5 * h_try
            ya, ypa = do_step(F, t, h2, y, yp)
            yb, ypb = do_step(F, t + h2, h2, ya, ypa)
            if not torch.all(torch.isfinite(yb)):
                err = float("inf")
            else:
                scale = atol + rtol * yb.abs()
                err = float(((yb - y1).abs() / scale).max())
            if err <= 1.0 and torch.all(torch.isfinite(yb)):
                break
            shrink = min(0.9 * max(err, 1e-3) ** (-0.2), 0.9)
            h_try = max(h_try * shrink, h_floor)
            if h_try <= h_floor * 1.01:
                break
        else:
            # out of retries: take the finest attempt even if err is large
            n_fallback += 1
        n_retries += attempt if 'attempt' in dir() else 0
        y, yp = yb, ypb
        t += h_try
        n_steps += 1
        ts.append(t)
        ys.append(y[0].clone().cpu())
        if not torch.all(torch.isfinite(y)):
            break
        if err < 1e-4:
            h = min(h * 5.0, h_max)
        elif err > 1.0:
            h = h
        else:
            h = min(h * max(1.05, min(4.0, 0.9 * err ** (-0.2))), h_max)
    return ts, ys, n_steps, n_retries, n_fallback


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
    print("n_sp=%d n_rx=%d t_end=%.6g s" % (len(keys), len(rps), t_end), flush=True)

    dydt, device, _ = rate_system(nu, rps, T, P)
    y0 = torch.tensor(init, dtype=torch.float64, device=device)[None, :]
    yp0 = dydt(y0)

    def F(t, y, yp):
        y1 = y if y.dim() == 1 else y[0]
        return yp - dydt(y1[None, :])[0]

    h_max = max(t_end / 32, 1e-18)
    h_start = max(t_end / 5000.0, 1e-18)
    t0 = time.time()
    try:
        ts, ys, n_steps, n_retries, n_fallback = adaptive_tr_bdf2(
            F, dydt, 0.0, float(t_end), y0, h_max, h_start=h_start,
            rtol=1e-4, atol=1e-10)
        wall = time.time() - t0
        Y = torch.stack(ys).numpy()
        fin = np.all(np.isfinite(Y), axis=1)
        n_fin = int(fin.sum())
        msg = "adaptive tr_bdf2: %d/%d finite  steps=%d retries=%d fallback=%d wall=%.1fs" % (
            n_fin, len(ts), n_steps, n_retries, n_fallback, wall)
        if fin[-1]:
            last = Y[-1]
            mn, mx, s = float(last.min()), float(last.max()), float(last.sum())
            phys = (mn >= -1e-6) and (mx <= 1 + 1e-6) and abs(s - 1) < 1e-3
            top = [(keys[k], round(float(last[k]), 5))
                   for k in np.argsort(-last)[:5]]
            msg += ("  PHYSICAL=%s min=%.2e max=%.2e sum=%.6f top=%s"
                    % (phys, mn, mx, s, top))
        else:
            fb = int(np.where(~fin)[0][0])
            msg += "  first-bad row %d t=%s" % (fb, ts[fb] if np.isfinite(ts[fb]) else 'nan')
        print(msg, flush=True)
    except Exception as e:
        import traceback
        print("adaptive EXC %s: %s" % (type(e).__name__, str(e)[:200]), flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
