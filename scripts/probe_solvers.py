"""Probe: take the real superminimal final mechanism (loop iteration 3) and
integrate the SAME (keys, nu, rps, init, T, P, t_end) with:
  (a) solve_tr_bdf2 fixed h = t_end/32   (current rmgpu code)
  (b) solve_bdf2      adaptive (h=None)
  (c) solve_tr_bdf2   adaptive (h=None)
Report: n_steps, wall time, finiteness, row-sum error, final top species.
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 /home/jackson/miniforge3/envs/rmgpu/bin/python scripts/probe_solvers.py
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

import rmgpu.main as M  # noqa: E402
from rmgpu.core.loop import CoreEdgeLoop, RunContext, LoopConfig, canonical_key  # noqa: E402
from rmgpu.reactor.simulator import (RateParam, characteristic_rate,  # noqa: E402
                                     forward_A_T, reverse_factor, _choose_device,
                                     R)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_loop():
    input_path = os.path.join(REPO, "examples", "superminimal.yaml")
    mi = M.load_input(input_path)
    databases = M._build_databases(mi)
    ml = M._build_ml()
    fam_sel = mi.database.kinetics_families
    families = M._build_families(fam_sel)
    seed_species = M._build_seed_species(mi)
    sm_sp, sm_rx, sm_sum = M._build_seed_mechanisms(mi, databases)
    T, P, imf, t_term, conv = M._build_reactor(mi)
    db_block = getattr(mi, "database", None)
    ctx = RunContext(
        databases=databases, ml=ml, families=families,
        seed_species=seed_species, initial_mole_fractions=imf,
        temperature=T, pressure=P,
        config=LoopConfig(tolerance_move_to_core=0.001,
                          tolerance_keep_in_edge=0.0, max_iterations=3),
        thermo_libraries=list(db_block.thermo_libraries or [])
        if db_block else None,
        reaction_libraries=[], termination_time=t_term,
        termination_conversion=conv)
    ctx.seed_mechanisms_species = sm_sp
    ctx.seed_mechanisms_reactions = sm_rx
    ctx.seed_mechanisms_summaries = sm_sum
    loop = CoreEdgeLoop(ctx)
    for sp in ctx.seed_species:
        sp.is_seed = True
        loop.model.add_species_to_core(sp)
        loop.species_by_key[canonical_key(sp.molecule)] = sp
        loop._thermo(sp, 0)
    return loop, ctx, T, P


def rate_system(keys, nu, rps, T, P):
    """Build dydt + F exactly like simulator.simulate_mole_fractions does."""
    device = _choose_device()
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
    A_T = torch.tensor([forward_A_T(rp, T) for rp in rps],
                       dtype=torch.float64, device=device)
    rev_t = torch.tensor([reverse_factor(rp, T, float(dnu_list[j]))
                          for j, rp in enumerate(rps)],
                         dtype=torch.float64, device=device)
    k_fwd = A_T * torch.pow(torch.tensor(c_tot, dtype=torch.float64, device=device),
                            n_react_t - 1.0)
    k_rev = A_T * rev_t * torch.pow(torch.tensor(c_tot, dtype=torch.float64, device=device),
                                    n_prod_t - 1.0)
    eps = 1e-30

    def rates(yv):
        batch = yv.shape[0]
        net = []
        for j in range(n_rx):
            if len(react_i[j]) > 0:
                y_r = torch.clamp(yv[:, react_i[j]], min=eps)
                f = torch.ones(batch, device=device)
                for k in range(len(react_i[j])):
                    f = f * (y_r[:, k] ** react_c[j][k])
            else:
                f = torch.ones(batch, device=device)
            if len(prod_i[j]) > 0:
                y_p = torch.clamp(yv[:, prod_i[j]], min=eps)
                rf = torch.ones(batch, device=device)
                for k in range(len(prod_i[j])):
                    rf = rf * (y_p[:, k] ** prod_c[j][k])
            else:
                rf = torch.ones(batch, device=device)
            net.append(k_fwd[j] * f - k_rev[j] * rf)
        return torch.stack(net, dim=1)

    def dydt(y):
        net = rates(y)
        out = torch.zeros_like(y)
        for j in range(n_rx):
            nj = net[:, j]
            for k in range(len(nu_i[j])):
                out[:, nu_i[j][k]] += nj * nu_c[j][k]
        dnu_sum = (net * dnu_t.unsqueeze(0)).sum(dim=1, keepdim=True)
        return out - y * dnu_sum

    return dydt, device


def report(name, ts, ys, wall, keys):
    finite = np.all(np.isfinite(ys))
    sums = ys.sum(axis=1)
    sumerr = float(np.max(np.abs(sums - 1.0))) if finite else float("nan")
    mx = float(np.max(np.abs(ys))) if finite else float("nan")
    last_top = []
    if finite:
        idx = np.argsort(-ys[-1])[:5]
    else:
        idx = np.argsort(-np.nan_to_num(ys[-1]))[:5]
    print(f"{name:38s} steps={len(ts):5d} wall={wall:6.1f}s finite={finite} "
          f"sum_err={sumerr:.3e} max|y|={mx:.3e} last_t={ts[-1]:.6g}")
    if finite:
        print(f"   final top5: {[(keys[i], round(float(ys[-1][i]), 6)) for i in idx]}")


def main():
    loop, ctx, T, P = build_loop()
    # run the loop to iteration 3 (same as the smoke run) to get the final sim
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
    t_end = 5.0 / char
    t_end = min(t_end, float(ctx.termination_time or 1e12))
    t_end = max(1e-9, min(t_end, 1e12))
    keys, nu, rps, y0 = sim["keys"], sim["nu"], sim["rps"], sim["init"]
    print(f"n_sp={len(keys)} n_rx={len(rps)} char={char:.6g} t_end={t_end:.6g} s")

    dydt, device = rate_system(keys, nu, rps, T, P)
    y0_t = torch.tensor(y0, dtype=torch.float64, device=device)[None, :]
    yp0 = dydt(y0_t)

    def F(t, y, yp):
        y1 = y if y.dim() == 1 else y[0]
        return yp - dydt(y1[None, :])[0]

    # (a) fixed-h TR-BDF2 (current code)
    h = max(t_end / 32, 1e-18)
    t0 = time.time()
    sol = torchdae.solve_tr_bdf2(F, (0.0, float(t_end)), y0_t, h=float(h), yp0=yp0)
    ts = sol.ts.cpu().numpy()
    ys = sol.ys.squeeze(1).cpu().numpy()
    report("(a) tr_bdf2 fixed h=t_end/32", ts, ys, time.time() - t0, keys)

    # (b) adaptive solve_bdf2
    t0 = time.time()
    try:
        sol = torchdae.solve_bdf2(F, (0.0, float(t_end)), y0_t, h=None, yp0=yp0,
                                  max_iter_for_events=500)
        ts = sol.ts.cpu().numpy()
        ys = sol.ys.squeeze(1).cpu().numpy()
        report("(b) bdf2 adaptive h=None", ts, ys, time.time() - t0, keys)
    except Exception as e:
        print(f"(b) bdf2 adaptive: EXC {type(e).__name__}: {str(e)[:150]}")

    # (c) adaptive tr_bdf2
    t0 = time.time()
    try:
        sol = torchdae.solve_tr_bdf2(F, (0.0, float(t_end)), y0_t, h=None, yp0=yp0)
        ts = sol.ts.cpu().numpy()
        ys = sol.ys.squeeze(1).cpu().numpy()
        report("(c) tr_bdf2 adaptive h=None", ts, ys, time.time() - t0, keys)
    except Exception as e:
        print(f"(c) tr_bdf2 adaptive: EXC {type(e).__name__}: {str(e)[:150]}")


if __name__ == "__main__":
    main()
