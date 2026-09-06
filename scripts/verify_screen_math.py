"""Standalone validation: the vectorized _screen mass-action + trapezoid in
rmgpu/core/loop.py must equal a naive per-reaction reference loop (incl.
zero-concentration reactants). Run:
    CUDA_VISIBLE_DEVICES= /home/jackson/miniforge3/envs/rmgpu/bin/python scripts/verify_screen_math.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmgpu.reactor.simulator import RateParam, forward_A_T, reverse_factor


def screen_reference(nu, rps, char, T, P, ys, times):
    """Naive per-reaction reference (the original loop semantics)."""
    c_tot = P / (8.314472 * T)
    sp_ratio = {}
    for j, rp in enumerate(rps):
        aT = forward_A_T(rp, T)
        n_react = sum(-m for m in nu[j] if m < 0)
        n_prod = sum(m for m in nu[j] if m > 0)
        rates = []
        for row in ys:
            pf = 1.0
            pr = 1.0
            for i, m in enumerate(nu[j]):
                y = max(0.0, row[i])
                if m < 0:
                    pf *= y ** (-m)
                elif m > 0:
                    pr *= y ** m
            k_f = aT * (c_tot ** (n_react - 1)) * pf
            k_r = (aT * reverse_factor(rp, T, float(n_prod - n_react))
                   * (c_tot ** (n_prod - 1)) * pr)
            rates.append(max(k_f, k_r))
        rates = np.array(rates)
        integral = float(np.sum(0.5 * (rates[:-1] + rates[1:]) * np.diff(times)))
        rr = (integral / (times[-1] - times[0])) / char
        for i, m in enumerate(nu[j]):
            if m != 0:
                sp_ratio[i] = max(sp_ratio.get(i, 0.0), rr)
    return sp_ratio


def screen_vectorized(nu, rps, char, T, P, ys, times):
    """Mirror of the current rmgpu/core/loop.py _screen computation."""
    c_tot = P / (8.314472 * T)
    nmat = np.asarray(nu, dtype=float)
    log_y = np.log(np.maximum(ys, 1e-30))
    n_react = np.where(nmat < 0, -nmat, 0.0).sum(axis=1)
    n_prod = np.where(nmat > 0, nmat, 0.0).sum(axis=1)
    A_T = np.array([forward_A_T(rp, T) for rp in rps])
    rev_t = np.array([reverse_factor(rp, T, float(n_prod[j] - n_react[j]))
                      for j, rp in enumerate(rps)])
    react_w = np.clip(-nmat, 0.0, None)
    prod_w = np.clip(nmat, 0.0, None)
    fwd = A_T[:, None] * (c_tot ** (n_react[:, None] - 1.0)) \
        * np.exp(react_w @ log_y.T)
    rev = A_T[:, None] * rev_t[:, None] \
        * (c_tot ** (n_prod[:, None] - 1.0)) \
        * np.exp(prod_w @ log_y.T)
    rates = np.maximum(fwd, rev)
    dt = np.diff(times)
    integral = np.sum(0.5 * (rates[:, :-1] + rates[:, 1:]) * dt[None, :], axis=1)
    rr_vec = (integral / (times[-1] - times[0])) / char
    rr_by_species = np.where(nmat != 0.0, rr_vec[:, None], -np.inf)
    sp_max = np.nanmax(rr_by_species, axis=0)
    return {i: (float(sp_max[i]) if np.isfinite(sp_max[i]) else 0.0)
            for i in range(nmat.shape[1])}


def main():
    ok = True
    for seed in (1, 2, 3):
        rng = np.random.RandomState(seed)
        n_sp, n_rx, n_steps = 12, 8, 7
        ys = np.abs(rng.rand(n_steps, n_sp)) * 0.1
        ys[2, 3] = 0.0  # zero-concentration reactant case
        times = np.linspace(0, 5, n_steps)
        nu = []
        rps = []
        for j in range(n_rx):
            v = np.zeros(n_sp)
            for _ in range(rng.randint(1, 3)):
                v[rng.randint(n_sp)] -= 1
            for _ in range(rng.randint(1, 3)):
                v[rng.randint(n_sp)] += 1
            nu.append(v.tolist())
            rps.append(RateParam(A=float(rng.uniform(1e6, 1e12)), n=0,
                                 Ea=float(rng.uniform(0, 2e4)), T0=1.0,
                                 dS=float(rng.uniform(-50, 50)),
                                 dH=float(rng.uniform(-5e4, 5e4)),
                                 reversible=True))
        T, P, char = 1350.0, 1e5, 1234.56
        new = screen_vectorized(nu, rps, char, T, P, ys, times)
        ref = screen_reference(nu, rps, char, T, P, ys, times)
        worst = 0.0
        for i in range(n_sp):
            a = new.get(i, 0.0)
            b = ref.get(i, 0.0)
            worst = max(worst, abs(a - b) / max(abs(a), abs(b), 1e-300))
        print(f"seed={seed} max rel diff = {worst:.3e}")
        ok = ok and worst < 1e-9
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
