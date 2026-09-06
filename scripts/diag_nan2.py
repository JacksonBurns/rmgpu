"""Diagnose the superminimal NaN from the WRITTEN core.yaml (no ML load):
dump per-reaction rate constants + integrate the final core at several step
counts, showing where/why it goes non-finite.
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 /home/jackson/miniforge3/envs/rmgpu/bin/python scripts/diag_nan2.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
import torch  # noqa: E402
import torchdae  # noqa: E402
import logging  # noqa: E402
logging.disable(logging.CRITICAL)

from rmgpu.molecule.molecule import Molecule  # noqa: E402
from rmgpu.reactor.simulator import (RateParam, characteristic_rate,  # noqa: E402
                                     forward_A_T, reverse_factor, R,
                                     _choose_device)
from rmgpu.units import Quantity  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "examples", "run_output_sm_smoke")


def build_from_artifact():
    art = yaml.safe_load(open(os.path.join(ROOT, "mechanism", "core.yaml")))
    core = art["core"]
    sp_map = {}
    for sp in core["species"]:
        sp_map[sp["label"]] = Molecule(smiles=sp["smiles"])
    rps, nu, keys = [], [], []
    seen = set()
    for r in core["reactions"]:
        p = r["rate"]["params"]
        rp = RateParam(A=p.get("A", 0.0), n=p.get("n", 0.0), Ea=p.get("Ea", 0.0),
                       T0=p.get("T0", 1.0), dS=p.get("dS", 0.0), dH=p.get("dH", 0.0),
                       family=r.get("family"), source=r.get("source", "ml"))
        rps.append(rp)
        for k in r["reactants"] + r["products"]:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    keys.sort()
    idx = {k: i for i, k in enumerate(keys)}
    nus = []
    for r in core["reactions"]:
        row = [0.0] * len(keys)
        for k in r["reactants"]:
            row[idx[k]] -= 1
        for k in r["products"]:
            row[idx[k]] += 1
        nus.append(row)
    run_doc = yaml.safe_load(open(os.path.join(ROOT, "run.yaml")))
    T = float(Quantity(run_doc["reactors"][0]["temperature"]["value"],
                       run_doc["reactors"][0]["temperature"]["unit"]).to_si())
    P = float(Quantity(run_doc["reactors"][0]["pressure"]["value"],
                       run_doc["reactors"][0]["pressure"]["unit"]).to_si())
    imf = run_doc["reactors"][0].get("initial_mole_fractions") or {}
    init = [float(imf.get(k, 0.0)) for k in keys]
    return keys, nus, rps, T, P, init


def rate_system(keys, nu, rps, T, P):
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
    keys, nu, rps, T, P, init = build_from_artifact()
    char = characteristic_rate(nu, rps, T, P, init)
    t_end = min(5.0 / char, 1.0e6)
    t_end = max(1e-9, min(t_end, 1e12))
    print(f"n_sp={len(keys)} n_rx={len(rps)} T={T} P={P:.3e} char={char:.6g} "
          f"t_end={t_end:.6g} s")

    dydt, device, c_tot = rate_system(keys, nu, rps, T, P)
    print("\n=== per-reaction rate constants (T=%.0f K, P=%.0f Pa) ===" % (T, P))
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
            flag = "  <-- HUGE (>1e38)"
        rkey = "+".join(keys[i] for i, m in enumerate(nu[j]) if m < 0)
        pkey = "+".join(keys[i] for i, m in enumerate(nu[j]) if m > 0)
        print(f"rxn {j:2d} {rkey} >> {pkey} | fam={rp.family} src={rp.source} "
              f"A={rp.A:.3e} Ea={rp.Ea:.0f} dS={rp.dS:.1f} dH={rp.dH:.0f} "
              f"rev={rt:.3e} kf={kf:.3e} kr={kr:.3e}{flag}")

    y0_t = torch.tensor(init, dtype=torch.float64, device=device)[None, :]
    yp0 = dydt(y0_t)

    def F(t, y, yp):
        y1 = y if y.dim() == 1 else y[0]
        return yp - dydt(y1[None, :])[0]

    for n in (32, 512):
        h = max(t_end / n, 1e-18)
        sol = torchdae.solve_tr_bdf2(F, (0.0, float(t_end)), y0_t, h=float(h), yp0=yp0)
        ts = sol.ts.cpu().numpy()
        ys = sol.ys.squeeze(1).cpu().numpy()
        fin = np.all(np.isfinite(ys), axis=1)
        n_fin = int(fin.sum())
        last_fin = np.where(fin)[0][-1] if n_fin else -1
        print(f"\n=== n={n}: {n_fin}/{len(ts)} rows finite ===")
        if last_fin >= 0:
            for i in (last_fin - 1, last_fin):
                if 0 <= i < len(ts):
                    row = ys[i]
                    top = np.argsort(-np.abs(np.nan_to_num(row)))[:5]
                    print(f"  row {i} t={ts[i]:.4g} sum={row.sum():.6f} "
                          f"top={[ (keys[k], round(float(row[k]), 8)) for k in top ]}")
        # find first non-finite
        if n_fin < len(ts):
            first_bad = np.where(~fin)[0][0]
            row = ys[first_bad]
            top = np.argsort(-np.abs(np.nan_to_num(row)))[:5]
            print(f"  FIRST BAD row {first_bad} t={ts[first_bad]:.4g}: "
                  f"{[ (keys[k], repr(row[k])) for k in top ]}")


if __name__ == "__main__":
    main()
