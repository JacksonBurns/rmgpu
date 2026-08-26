"""Tests for rmgpu.kinetics.models.

Compares the pure-Python Arrhenius implementation against RMG-Py's Cython
implementation for several parameter combinations and temperatures.  Also checks
the ArrheniusEP activation-energy clamping behaviour and the reverse-rate
helper on a toy reaction.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from rmgpu.kinetics.models import (
    R,
    Arrhenius,
    ArrheniusEP,
    ArrheniusBM,
    Chebyshev,
    Eckart,
    Lindemann,
    Marcus,
    PDepArrhenius,
    ThirdBody,
    Troe,
    Wigner,
    generate_reverse_rate_coefficient,
    make_rate_model,
)

# Reference values computed from RMG-Py's Arrhenius implementation.
# These use the same parameter sets listed in the test below.
RMGPY_CASES = [
    # (A, n, Ea, T0, T_grid, expected_k_grid)
    (1e13, 0.0, 50e3, 1.0, np.linspace(300, 1500, 12),
     np.array([1.969728958085e+04, 4.129505775261e+06, 9.118418801753e+07,
               6.862071961868e+08, 2.839727779313e+09, 8.145512692727e+09,
               1.836353635333e+10, 3.504097516609e+10, 5.929085342468e+10,
               9.173195791591e+10, 1.325326751034e+11, 1.815019494281e+11])),
    (5e5, 1.0, 10e3, 1.0, np.linspace(300, 1500, 12),
     np.array([2.722529241422e+06, 1.081357877922e+07, 2.543525417342e+07,
               4.610175554488e+07, 7.189814408542e+07, 1.019163713782e+08,
               1.353810512309e+08, 1.716636797252e+08, 2.102638294554e+08,
               2.507844269521e+08, 2.929094226593e+08, 3.363856975502e+08])),
    (2e7, -0.5, 25e3, 300.0, np.linspace(300, 1500, 12),
     np.array([8.876325722020e+02, 1.100600606068e+04, 4.595251709944e+04,
               1.145750497011e+05, 2.151210015143e+05, 3.400201884329e+05,
               4.804746461414e+05, 6.287559771461e+05, 7.789076356094e+05,
               9.266969653537e+05, 1.069307793741e+06, 1.204996097680e+06])),
]


def test_arrhenius_matches_rmGPY():
    for A, n, Ea, T0, T_grid, expected in RMGPY_CASES:
        model = Arrhenius(A=A, n=n, Ea=Ea, T0=T0)
        k = np.array([model.get_rate_coefficient(T) for T in T_grid])
        np.testing.assert_allclose(k, expected, rtol=1e-4, atol=0.0)


def test_arrhenius_with_T0():
    """Changing T0 changes the rate expression but not k at T0."""
    model1 = Arrhenius(A=1e13, n=0.5, Ea=30e3, T0=1.0)
    model2 = Arrhenius(A=1e13 * np.sqrt(500), n=0.5, Ea=30e3, T0=500.0)
    assert np.isclose(model1.get_rate_coefficient(500.0),
                      model2.get_rate_coefficient(500.0), rtol=1e-8)


def test_arrhenius_change_t0():
    model = Arrhenius(A=1e13, n=0.5, Ea=30e3, T0=1.0)
    k_at_T0_old = model.get_rate_coefficient(500.0)
    model.change_t0(500.0)
    assert np.isclose(model.A, 1e13 * np.sqrt(500.0), rtol=1e-8)
    assert np.isclose(model.get_rate_coefficient(500.0), k_at_T0_old, rtol=1e-8)


def test_arrhenius_ep_activation_energy_clamping():
    # E0 positive, dHrxn negative enough to drive Ea below zero -> clamp to 0.
    aep = ArrheniusEP(A=1e13, alpha=0.5, E0=10e3)
    assert np.isclose(aep.get_activation_energy(-30e3), 0.0, atol=1e-12)

    # dHrxn positive but smaller than Ea -> no change.
    assert np.isclose(aep.get_activation_energy(15e3), 0.5 * 15e3 + 10e3, atol=1e-12)

    # dHrxn positive and larger than Ea -> clamp to dHrxn.
    assert np.isclose(aep.get_activation_energy(30e3), 30e3, atol=1e-12)


def test_arrhenius_ep_rate_coefficient():
    aep = ArrheniusEP(A=1e13, n=1.0, alpha=0.5, E0=10e3)
    T = 1000.0
    dHrxn = -20e3
    expected = aep.A * T ** aep.n * np.exp(-aep.get_activation_energy(dHrxn) / (R * T))
    assert np.isclose(aep.get_rate_coefficient(T, dHrxn), expected, rtol=1e-8)


def test_arrhenius_ep_to_arrhenius():
    aep = ArrheniusEP(A=1e13, n=0.5, alpha=0.5, E0=10e3, Tmin=300.0, Tmax=2000.0)
    arr = aep.to_arrhenius(15e3)
    assert np.isclose(arr.Ea, 0.5 * 15e3 + 10e3, atol=1e-12)
    assert np.isclose(arr.T0, 1.0, atol=1e-12)
    assert arr.Tmin == aep.Tmin
    assert arr.Tmax == aep.Tmax


def test_make_rate_model_registry():
    arr = make_rate_model("arrhenius", A=1e13, n=0.0, Ea=50e3)
    assert isinstance(arr, Arrhenius)
    aep = make_rate_model("arrhenius_ep", A=1e13, alpha=0.5, E0=10e3)
    assert isinstance(aep, ArrheniusEP)
    with pytest.raises(KeyError):
        make_rate_model("unknown", A=1.0)


def test_reverse_rate_coefficient():
    k_forward = 1.0e10
    dH_rxn = 50e3
    dS_rxn = 100.0
    T = 500.0
    k_rev = generate_reverse_rate_coefficient(k_forward, dH_rxn, dS_rxn, T)
    expected = k_forward * np.exp((dH_rxn - T * dS_rxn - dH_rxn) / (R * T))
    assert np.isclose(k_rev, expected, rtol=1e-8)


def test_third_body_evaluates_low_pressure_limit():
    k0 = Arrhenius(A=1e5, n=0.0, Ea=0.0, T0=1.0)
    tb = ThirdBody(arrheniusLow=k0)
    T, P = 300.0, 1e5
    k = tb.get_rate_coefficient(T, P)
    expected = k0.get_rate_coefficient(T) * P / (R * T)
    np.testing.assert_allclose(k, expected, rtol=1e-10)


def test_lindemann_evaluates_intermediate_pressure():
    k0 = Arrhenius(A=1e5, n=0.0, Ea=0.0, T0=1.0)
    kinf = Arrhenius(A=1e13, n=0.0, Ea=0.0, T0=1.0)
    lind = Lindemann(arrheniusLow=k0, arrheniusHigh=kinf)
    T, P = 300.0, 1e4
    k = lind.get_rate_coefficient(T, P)
    C = P / (R * T)
    expected = kinf.get_rate_coefficient(T) * (k0.get_rate_coefficient(T) * C / kinf.get_rate_coefficient(T)) / (
        1.0 + k0.get_rate_coefficient(T) * C / kinf.get_rate_coefficient(T)
    )
    np.testing.assert_allclose(k, expected, rtol=1e-10)


def test_troe_evaluates_broadened_falloff():
    k0 = Arrhenius(A=1e5, n=0.0, Ea=0.0, T0=1.0)
    kinf = Arrhenius(A=1e13, n=0.0, Ea=0.0, T0=1.0)
    troe = Troe(arrheniusLow=k0, arrheniusHigh=kinf, alpha=0.5, T1=500.0, T2=0.0, T3=1000.0)
    T, P = 300.0, 1e4
    k = troe.get_rate_coefficient(T, P)
    assert k > 0


def test_chebyshev_evaluates_pdep_grid():
    coeffs = np.array([[0.5, -0.1], [0.0, 0.2]])
    chb = Chebyshev(
        coeffs=coeffs,
        Tmin=300.0,
        Tmax=1000.0,
        Pmin=1e3,
        Pmax=1e6,
    )
    T, P = 500.0, 1e4
    k = chb.get_rate_coefficient(T, P)
    assert k > 0


def test_pdep_arrhenius_storage_and_evaluation():
    high = Arrhenius(A=1e13, n=0.0, Ea=0.0, T0=1.0)
    model = PDepArrhenius(A=1e5, n=0.0, Ea=0.0, T0=1.0, Pmin=1e3, Pmax=1e6, highPlimit=high)
    T, P = 500.0, 1e4
    k = model.get_rate_coefficient(T, P)
    assert k > 0
    k_max = model.get_rate_coefficient(T, 1e7)
    assert np.isclose(k_max, high.get_rate_coefficient(T), rtol=1e-8)


def test_marcus_rate_model_evaluates_forward_barrier():
    model = Marcus(A=1e13, n=0.0, Ea=0.0, T0=1.0, lambda_=20e3)
    T, dG = 300.0, 5e3
    k = model.get_rate_coefficient(T, dG)
    assert k > 0
    assert model.get_rate_coefficient(T, -5e3) == 0.0

def test_marcus_stores_and_uses_dG():
    model = Marcus(A=1e13, n=0.0, Ea=0.0, T0=1.0, lambda_=20e3, dG=5e3)
    T = 300.0
    k = model.get_rate_coefficient(T)
    assert k > 0
    assert model.get_rate_coefficient(T, dG=5e3) > 0


def test_wigner_tunneling_factor():
    wigner = Wigner(frequency=1000.0)
    factor = wigner.calculate_tunneling_factor(300.0)
    assert factor > 1.0


def test_eckart_tunneling_function_shape():
    eckart = Eckart(frequency=1000.0, E0_reac=0.0, E0_TS=50e3, E0_prod=-10e3)
    Elist = np.arange(0.0, 120e3, 100.0)
    kappa = eckart.calculate_tunneling_function(Elist)
    assert kappa.shape == Elist.shape
    assert np.all(kappa >= 0.0)
    assert np.all(kappa <= 1.0 + 1e-9)


def test_arrhenius_bm_storage_only():
    bm = ArrheniusBM(A=1e13, n=0.0, Ea=50e3, T0=1.0)
    with pytest.raises(NotImplementedError):
        bm.get_rate_coefficient(300.0)


def test_make_rate_model_new_models():
    assert isinstance(make_rate_model("third_body", arrheniusLow=Arrhenius(A=1.0)), ThirdBody)
    assert isinstance(make_rate_model("lindemann", arrheniusLow=Arrhenius(A=1.0), arrheniusHigh=Arrhenius(A=1.0)), Lindemann)
    assert isinstance(make_rate_model("troe", arrheniusLow=Arrhenius(A=1.0), arrheniusHigh=Arrhenius(A=1.0)), Troe)
    assert isinstance(make_rate_model("chebyshev", coeffs=np.array([[0.0]]), Tmin=300.0, Tmax=1000.0, Pmin=1e3, Pmax=1e6), Chebyshev)
    assert isinstance(make_rate_model("pdep_arrhenius", A=1e5), PDepArrhenius)
    assert isinstance(make_rate_model("marcus", A=1e13, lambda_=20e3), Marcus)
    assert isinstance(make_rate_model("arrhenius_bm", A=1e13), ArrheniusBM)
    assert isinstance(make_rate_model("wigner", frequency=1000.0), Wigner)
    assert isinstance(make_rate_model("eckart", frequency=1000.0, E0_reac=0.0, E0_TS=50e3, E0_prod=-10e3), Eckart)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
