import pytest
from rmgpu.reactor.simulator import RateParam, simulate_mole_fractions

def test_sparse_simulation_mole_fraction_sum():
    keys = ["A", "B", "C"]
    nu = [
        [-1, 1, 0],  # A -> B
        [0, -1, 1],  # B -> C
    ]
    rps = [
        RateParam(A=1e10, n=0, Ea=0, T0=298.15, dS=0, dH=0, reversible=False),
        RateParam(A=1e10, n=0, Ea=0, T0=298.15, dS=0, dH=0, reversible=False),
    ]
    T = 1000.0
    P = 1e5
    y0 = [0.5, 0.5, 0.0]
    t_end = 1e-6
    h = 1e-7
    res = simulate_mole_fractions(keys, nu, rps, T, P, y0, t_end, h)
    for ys in res.ys:
        s = sum(ys)
        assert abs(s - 1.0) < 1e-4, f"Sum {s} deviates from 1"
