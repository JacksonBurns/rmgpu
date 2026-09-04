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

    nu_t = torch.tensor(nu, dtype=torch.float64, device=device)  # (rxn, sp)
    neg_nu = (-nu_t).clamp(min=0).T      # (sp, rxn) forward powers
    pos_nu = nu_t.clamp(min=0).T         # (sp, rxn) reverse powers
    n_react = (-nu_t).clamp(min=0).sum(dim=1)   # (rxn,) reactant count
    n_prod = nu_t.clamp(min=0).sum(dim=1)       # (rxn,) product count
    dnu_t = nu_t.sum(dim=1)                     # (rxn,) mole-count change

    c_tot = P / (R * T)                    # mol/m^3, constant
    A_T = torch.tensor([forward_A_T(rp, T) for rp in rps],
                       dtype=torch.float64, device=device)
    rev_t = torch.tensor([reverse_factor(rp, T, float(dnu_t[j].item()))
                          for j, rp in enumerate(rps)],
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
        # species_i rate = sum_j (nu_ij - y_i*dnu_j) * net_j -> (batch, sp)
        # The -y_i*dnu_j term is the constant-T/P dilution (see module doc):
        # it makes sum_i dy_i/dt = 0 exactly, so the mole-fraction vector
        # stays closed for mole-count-changing reactions.
        net = rates(y)                     # (batch, rxn)
        dnu = net @ dnu_t                  # (batch,) sum_j dnu_j net_j
        return net @ nu_t - y * dnu.unsqueeze(1)

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
