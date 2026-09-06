"""Probe the LIVE superminimal iteration-3 sim (11 spc / 15 rxn, the one that
NaNs at t~1.5e5 with TR-BDF2 n=32):
  1) dump all 15 reactions' rate constants (flag huge/non-finite);
  2) integrate with TR-BDF2 n=32 (current) / n=512 / Radau IIA5 n=32 / n=512;
  3) report finiteness + where the first non-finite row appears.
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 /home/jackson/miniforge3/envs/rmgpu/bin/python scripts/probe_live_sm.py
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
from rmgpu.reactor.simulator import (characteristic_rate, forward_A_T,  # noqa: E402
                                     reverse_factor, R, _choose_device)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_loop():
    mi = M.load_input(os.path.join(REPO, "examples", "superminimal.yaml"))
    databases = M._build_databases(mi)
    ml = M._build_ml()
    families = M._build_families(mi.database.kinetics_families)
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
        thermo_libraries=list(db_block.thermo_libraries or []) if db_block else None,
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


def rate_system(nu, rps, T, P):
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

    return dydt, device, c_tot


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
    print(f"n_sp={len(keys)} n_rx={len(rps)} char={char:.6g} t_end={t_end:.6g} s", flush=True)

    _, _, c_tot = rate_system(nu, rps, T, P)
    print("\n=== 15 live reactions (T=%.0f K) ===" % T, flush=True)
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
        elif max(abs(np.log10(kf + 1e-300)), abs(np.log10(kr + 1e-300))) > 38:
            flag = "  <-- HUGE(>1e38)"
        rkey = "+".join(keys[i] for i, m in enumerate(nu[j]) if m < 0)
        pkey = "+".join(keys[i] for i, m in enumerate(nu[j]) if m > 0)
        print(f"  rxn{j:2d} {rkey}>>{pkey} fam={rp.family:20s} src={rp.source:7s} "
              f"A={rp.A:.3e} Ea={rp.Ea:9.0f} dS={rp.dS:8.1f} dH={rp.dH:9.0f} "
              f"rev={rt:.3e} kf={kf:.3e} kr={kr:.3e}{flag}", flush=True)

    dydt, device, _ = rate_system(nu, rps, T, P)
    y0_t = torch.tensor(init, dtype=torch.float64, device=device)[None, :]
    yp0 = dydt(y0_t)

    def F(t, y, yp):
        y1 = y if y.dim() == 1 else y[0]
        return yp - dydt(y1[None, :])[0]

    for name, solver, n in (("tr_bdf2 n=32  ", torchdae.solve_tr_bdf2, 32),
                            ("tr_bdf2 n=512 ", torchdae.solve_tr_bdf2, 512),
                            ("radau_iia5 n=32", torchdae.solve_radau_iia5, 32),
                            ("radau_iia5 n=512", torchdae.solve_radau_iia5, 512)):
        h = max(t_end / n, 1e-18)
        try:
            sol = solver(F, (0.0, float(t_end)), y0_t, h=float(h), yp0=yp0)
            ts = sol.ts.cpu().numpy()
            ys = sol.ys.squeeze(1).cpu().numpy()
            fin = np.all(np.isfinite(ys), axis=1)
            n_fin = int(fin.sum())
            msg = f"{name}: {n_fin}/{len(ts)} finite"
            if n_fin < len(ts):
                fb = np.where(~fin)[0][0]
                msg += f" first-bad row {fb} t={ts[fb]:.4g}"
            elif fin[-1]:
                s = ys[-1].sum()
                msg += f" final_sum={s:.5f} top={[(keys[k], round(float(ys[-1][k]), 4)) for k in np.argsort(-ys[-1])[:4]]}"
            print(msg, flush=True)
        except Exception as e:
            print(f"{name}: EXC {type(e).__name__}: {str(e)[:120]}", flush=True)


if __name__ == "__main__":
    main()
