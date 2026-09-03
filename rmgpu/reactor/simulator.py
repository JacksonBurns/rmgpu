"""torchdae ODE integration for the mechanism simulation (job-06 loop).

State = the closed mole-fraction vector ``y`` (sums to 1). Constant T/P ideal
gas, total molar concentration ``c = P/(R T)`` constant, ``c_i = y_i * c``.
The mole balance ``d(c_i)/dt = sum_j nu_ij (fwd_j - rev_j)`` with ``c_i =
y_i c`` (c constant) gives the mole-fraction ODE

    dy_i/dt = sum_j nu_ij ( f_j - r_j )
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


def reverse_factor(rp: RateParam) -> float:
    """k_rev/k_fwd = exp(-dS_rxn/R) (thermodynamic consistency)."""
    if not rp.reversible:
        return 0.0
    return _safe_exp(-rp.dS / R)


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

    nu_t = torch.tensor(nu, dtype=torch.float64, device=device)  # (rxn, sp)
    neg_nu = (-nu_t).clamp(min=0).T      # (sp, rxn) forward powers
    pos_nu = nu_t.clamp(min=0).T         # (sp, rxn) reverse powers
    n_react = (-nu_t).clamp(min=0).sum(dim=1)   # (rxn,) reactant count
    n_prod = nu_t.clamp(min=0).sum(dim=1)       # (rxn,) product count

    c_tot = P / (R * T)                    # mol/m^3, constant
    A_T = torch.tensor([forward_A_T(rp, T) for rp in rps],
                       dtype=torch.float64, device=device)
    rev_t = torch.tensor([reverse_factor(rp) for rp in rps],
                         dtype=torch.float64, device=device)
    cfwd = torch.pow(torch.tensor(c_tot, dtype=torch.float64, device=device),
                     (n_react - 1.0))
    crev = torch.pow(torch.tensor(c_tot, dtype=torch.float64, device=device),
                     (n_prod - 1.0))
    k_fwd = A_T * cfwd
    k_rev = A_T * rev_t * crev

    def rates(yv):
        # yv: (n_batch, n_sp). neg_nu/pos_nu: (sp, rxn). k_*: (rxn,).
        # Returns (n_batch, n_rx) net rates.
        yb = yv.unsqueeze(2)                     # (batch, sp, 1)
        fwd = yb.pow(neg_nu.unsqueeze(0)).prod(dim=1) * k_fwd
        rev = yb.pow(pos_nu.unsqueeze(0)).prod(dim=1) * k_rev
        return fwd - rev                          # (batch, rxn)

    def dydt(y):
        # species_i rate = sum_j nu_ij * net_j -> (batch, rxn) @ (rxn, sp)
        return rates(y) @ nu_t                    # (n_batch, n_sp)

    y0_t = torch.tensor(y0, dtype=torch.float64, device=device)[None, :]

    def F(t, y, yp):
        # torchdae calls F with y/yp 1-D (functorch Jacobian) or 2-D
        # (batch=1, IC validation + integration). Normalize to the state
        # vector and keep the output matching yp's dimensionality.
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
