"""torchdae ODE integration for the mechanism simulation (job-06 loop).

State = the closed mole-fraction vector ``y`` (sums to 1). Constant T/P ideal
gas (isobaric reactor): the total molar concentration ``c = P/(R T)`` is
constant, ``c_i = y_i * c``. The mole balance ``d(c_i)/dt = sum_j nu_ij
(f_j - r_j)`` with ``c_i = y_i c`` and constant ``c`` gives

    dy_i/dt = sum_j nu_ij ( f_j - r_j )

BUT that is the CONSTANT-VOLUME form. At constant T AND P the reactor
VOLUME changes as the total mole number changes (dV/V = dn_tot/n_tot),
which adds a dilution term. In concentration form
d(c_i)/dt = c * sum_j (nu_ij - y_i * dnu_j) (f_j - r_j), dnu_j = sum_i
nu_ij, so the correct constant-T/P mole-fraction ODE is

    dy_i/dt = sum_j (nu_ij - y_i * dnu_j) ( f_j - r_j )

The dilution term makes the mole-fraction sum a conserved invariant
(sum_i dy_i/dt = 0 EXACTLY), which is what makes "the state is a closed
mole-fraction vector (sums to 1)" true. Without it, any mechanism with a
mole-count-changing reaction (every association/dissociation - all of
GRI-Mech3's termolecular/combination chemistry) drifts off sum 1
(measured: up to 13.5% off for a single 2A<=>B reaction over one
characteristic timescale; ~1.2% over the c3h4 seed mechanism).
job-06/step-07 added this term (verified: sum conserved to 4e-15 for
mole-count-changing reactions).

    f_j = A_j(T) * c^{n_react_j - 1} * prod_i y_i^{max(-nu_ij,0)}
    r_j = A_j(T) * rev_j * c^{n_prod_j - 1} * prod_i y_i^{max(nu_ij,0)}

where ``A_j(T) = A (T/T0)^n exp(-Ea/RT)`` (SI, from the rate model),
``n_react_j`` / ``n_prod_j`` are the reactant/product piece counts of
reaction j, and ``rev_j = exp(-dS_rxn/R)`` is the thermodynamic-consistency
reverse factor (RMG-Py: ``k_rev = k_fwd * exp((dG-dH)/RT)``; with
``dG = dH - T dS`` that ratio is ``exp(-dS/R)``).

The m^3<->cm^3 and mol/m^3<->mol/cm^3 factors in the c^{n-1} term cancel
against RMG-Py's CGS convention, which is what the job-02 rate round-trip
(1.4e-14) validates. Backend is torchdae TR-BDF2 (the single reactor
backend; PLAN 13 risk 3). The DAE residual is the ODE constraint
``F(t, y, yp) = yp - f(y)`` and handles the 1-D y (functorch Jacobian) and
the 2-D y (integration) shapes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torchdae
torch.set_default_dtype(torch.float32)

from rmgpu.kinetics.models import R  # 8.314472 (RMG-Py parity)


@dataclass
class RateParam:
    """SI Arrhenius rate parameters for one reaction (the forward, as
    written) plus the data needed for the thermodynamic reverse factor."""
    A: float              # SI m^{n_react-1} mol^{1-n_react} s^-1
    n: float
    Ea: float             # J/mol
    T0: float
    dS: float             # J/(mol K), reaction entropy change (prod - react)
    dH: float = 0.0       # J/mol, reaction enthalpy change (prod - react) at
                          # 298.15 K (Hf298 sum difference). Used in the
                          # thermodynamically consistent reverse factor (see
                          # reverse_factor). 0.0 when participant thermo is
                          # unavailable.
    reversible: bool = True
    family: Optional[str] = None
    template: Optional[List[str]] = None
    degeneracy: float = 1.0
    source: str = "ml"    # 'ml' | 'library'


@dataclass
class SimResult:
    times: List[float]
    ys: List[List[float]]          # [t][species] mole fractions

    def mole_fractions(self, keys) -> dict:
        return {k: [row[i] for row in self.ys] for i, k in enumerate(keys)}


def _safe_exp(x: float) -> float:
    if x > 700.0:
        return math.exp(700.0)
    if x < -700.0:
        return 0.0
    return math.exp(x)


def reverse_factor(rp: RateParam, T: float, dnu: float) -> float:
    """k_rev/k_fwd = 1/K_c(T), the thermodynamically consistent reverse factor.

    K_c(T) is the concentration-basis equilibrium constant
        K_c(T) = K_o * (P0/(R T))**dnu,  K_o = exp(-dG0/(R T)),
        dG0 = dH0 - T*dS0   (van 't Hoff, dCp neglected; dH0/dS0 at 298.15 K).
    Hence  k_rev/k_fwd = 1/K_c = exp(dG0/(R T)) * (R T/P0)**dnu.

    The PREVIOUS implementation used exp(-dS/R) (entropy only), which omitted
    the exp(dH0/(R T)) and (R T/P0)**dnu factors. For exothermic reactions
    (dH0 < 0) that factor is huge, so the reverse rate was overstated by up to
    ~30 orders of magnitude (measured on GRI-Mech3: 188/250 reversible
    reactions off by >100x). That drove wild dissociation of stable products
    (e.g. H2 -> H + H) and made the c3h4 profile non-physical. job-06/step-07
    root-caused the c3h4 RED to this and fixed it. See the step report.
    """
    if not rp.reversible:
        return 0.0
    P0 = 1.0e5  # standard-state pressure (Pa) = 1 bar, matches the thermo data
    dG0 = rp.dH - T * rp.dS
    return _safe_exp(dG0 / (R * T)) * ((R * T) / P0) ** dnu


def forward_A_T(rp: RateParam, T: float) -> float:
    """A_j(T) = A (T/T0)^n exp(-Ea/RT), SI."""
    arg = -rp.Ea / (R * T)
    if arg < -700.0:
        a = 0.0
    else:
        a = rp.A * (T / rp.T0) ** rp.n * math.exp(arg)
    return a


def _choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simulate_mole_fractions(
    keys: List[str],
    nu: List[List[float]],
    rps: List[RateParam],
    T: float,
    P: float,
    y0: List[float],
    t_end: float,
    h: float,
) -> SimResult:
    """Integrate the closed mole-fraction ODE with torchdae TR-BDF2."""
    device = _choose_device()
    n_sp = len(keys)
    n_rx = len(rps)
    if n_rx == 0:
        return SimResult(times=[0.0, float(t_end)],
                         ys=[list(y0), list(y0)])

    # Build sparse per-reaction species index lists to avoid dense (n_sp, n_rx) tensors
    reactant_idx_list = []
    reactant_coeff_list = []
    product_idx_list = []
    product_coeff_list = []
    nu_idx_list = []
    nu_coeff_list = []
    n_react_list = []
    n_prod_list = []
    dnu_list = []

    for j in range(n_rx):
        nu_j = nu[j]
        r_idx = []
        r_coeff = []
        p_idx = []
        p_coeff = []
        nu_idx = []
        nu_coeff = []
        n_react = 0
        n_prod = 0
        dnu = 0.0
        for i, c in enumerate(nu_j):
            if c < 0:
                r_idx.append(i)
                coeff = -c
                r_coeff.append(coeff)
                n_react += coeff
                dnu += c
                nu_idx.append(i)
                nu_coeff.append(c)
            elif c > 0:
                p_idx.append(i)
                p_coeff.append(c)
                n_prod += c
                dnu += c
                nu_idx.append(i)
                nu_coeff.append(c)
        reactant_idx_list.append(torch.tensor(r_idx, dtype=torch.long, device=device))
        reactant_coeff_list.append(torch.tensor(r_coeff, dtype=torch.float32, device=device))
        product_idx_list.append(torch.tensor(p_idx, dtype=torch.long, device=device))
        product_coeff_list.append(torch.tensor(p_coeff, dtype=torch.float32, device=device))
        nu_idx_list.append(torch.tensor(nu_idx, dtype=torch.long, device=device))
        nu_coeff_list.append(torch.tensor(nu_coeff, dtype=torch.float32, device=device))
        n_react_list.append(n_react)
        n_prod_list.append(n_prod)
        dnu_list.append(dnu)

    n_react_t = torch.tensor(n_react_list, dtype=torch.float32, device=device)
    n_prod_t = torch.tensor(n_prod_list, dtype=torch.float32, device=device)
    dnu_t = torch.tensor(dnu_list, dtype=torch.float32, device=device)

    c_tot = P / (R * T)
    A_T = torch.tensor([forward_A_T(rp, T) for rp in rps],
                       dtype=torch.float32, device=device)
    rev_t = torch.tensor([reverse_factor(rp, T, float(dnu_list[j]))
                          for j, rp in enumerate(rps)],
                         dtype=torch.float32, device=device)

    k_fwd = A_T * torch.pow(torch.tensor(c_tot, dtype=torch.float32, device=device),
                             n_react_t - 1.0)
    k_rev = A_T * rev_t * torch.pow(torch.tensor(c_tot, dtype=torch.float32, device=device),
                                     n_prod_t - 1.0)

    eps = 1e-30

    def rates(yv):
        # yv: (batch, n_sp)
        batch = yv.shape[0]
        net = []
        for j in range(n_rx):
            # forward
            idx_r = reactant_idx_list[j]
            coeff_r = reactant_coeff_list[j]
            if len(idx_r) > 0:
                y_r = torch.clamp(yv[:, idx_r], min=eps)
                fwd_factor = torch.ones(batch, device=device)
                for k in range(len(idx_r)):
                    fwd_factor = fwd_factor * (y_r[:, k] ** coeff_r[k])
            else:
                fwd_factor = torch.ones(batch, device=device)
            fwd = k_fwd[j] * fwd_factor

            # reverse
            idx_p = product_idx_list[j]
            coeff_p = product_coeff_list[j]
            if len(idx_p) > 0:
                y_p = torch.clamp(yv[:, idx_p], min=eps)
                rev_factor = torch.ones(batch, device=device)
                for k in range(len(idx_p)):
                    rev_factor = rev_factor * (y_p[:, k] ** coeff_p[k])
            else:
                rev_factor = torch.ones(batch, device=device)
            rev = k_rev[j] * rev_factor

            net_j = fwd - rev
            net.append(net_j)
        return torch.stack(net, dim=1)

    def dydt(y):
        net = rates(y)  # (batch, n_rx)
        # sparse scatter-add for net @ nu
        dydt_vals = torch.zeros_like(y)
        for j in range(n_rx):
            net_j = net[:, j]
            idxs = nu_idx_list[j]
            coeffs = nu_coeff_list[j]
            if len(idxs) > 0:
                for k in range(len(idxs)):
                    dydt_vals[:, idxs[k]] += net_j * coeffs[k]
        # dilution term
        dnu_sum = (net * dnu_t.unsqueeze(0)).sum(dim=1, keepdim=True)
        dydt_vals = dydt_vals - y * dnu_sum
        return dydt_vals

    y0_t = torch.tensor(y0, dtype=torch.float32, device=device)[None, :]

    def F(t, y, yp):
        y1 = y if y.dim() == 1 else y[0]
        return yp - dydt(y1[None, :])[0]

    yp0 = dydt(y0_t)
    sol = torchdae.solve_tr_bdf2(F, (0.0, float(t_end)), y0_t, h=float(h),
                                 yp0=yp0)
    ts = sol.ts.cpu().numpy().tolist()
    ys = sol.ys.squeeze(1).cpu().numpy().tolist()
    return SimResult(times=ts, ys=ys)


def characteristic_rate(nu: List[List[float]], rps: List[RateParam], T: float,
                        P: float, y0: List[float]) -> float:
    """The maximum forward reaction rate (mol/m^3/s) at the initial state -
    the characteristic rate RMG uses to set the simulation timescale and
    screening thresholds (base.pyx)."""
    n_sp = len(y0)
    c_tot = P / (R * T)
    maxr = 0.0
    for j, rp in enumerate(rps):
        aT = forward_A_T(rp, T)
        n_react = sum(-m for m in nu[j] if m < 0)
        prod = 1.0
        for i in range(n_sp):
            m = nu[j][i]
            if m < 0:
                prod *= max(0.0, y0[i]) ** (-m)
        r = aT * (c_tot ** (n_react - 1)) * prod
        maxr = max(maxr, r)
    return maxr
