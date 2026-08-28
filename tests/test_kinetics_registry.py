"""Tests for the rate registry and thermodynamic consistency."""

import numpy as np
import pytest

from rmgpu.kinetics.models import (
    Arrhenius,
    Lindemann,
    Troe,
    Chebyshev,
    PDepArrhenius,
    Marcus,
    Wigner,
    Eckart,
    RateRegistry,
    generate_reverse_rate_coefficient,
    generate_reverse_rate_coefficient_with_thermo,
)
from rmgpu.data.thermo import Wilhoit


class DummyThermo:
    """Mock thermo model for testing."""
    def __init__(self, H298=0, S298=0):
        self.H298 = H298
        self.S298 = S298

    def get_enthalpy(self, T):
        return self.H298

    def get_entropy(self, T):
        return self.S298


def test_wigner_factor_matches_rmgpy():
    """Wigner tunneling factor should match RMG-Py implementation."""
    # Test with frequency = 1000 cm^-1
    freq = 1000
    T = 300
    
    # RMG-Py formula
    c = 299792458.0  # m/s
    h = 6.62607015e-34  # J*s
    kB = 1.380649e-23  # J/K
    
    frequency = abs(freq) * c * 100.0
    factor = h * frequency / (kB * T)
    expected = 1.0 + factor**2 / 24.0
    
    # Our implementation
    wigner = Wigner(frequency=freq)
    actual = wigner.calculate_tunneling_factor(T)
    
    assert np.isclose(actual, expected, rtol=1e-10)


def test_eckart_tunneling_factor():
    """Eckart tunneling factor should be calculated correctly."""
    # Test with typical parameters
    freq = 1000  # cm^-1
    E0_reac = 0.0  # J/mol
    E0_TS = 1e6  # J/mol (1000 kJ/mol)
    E0_prod = 5e5  # J/mol
    
    eckart = Eckart(
        frequency=freq,
        E0_reac=E0_reac,
        E0_TS=E0_TS,
        E0_prod=E0_prod
    )
    
    T = 300  # K
    kappa = eckart.calculate_tunneling_factor(T)
    
    # Should be positive
    assert kappa > 0


def test_reverse_rate_coefficient_basic():
    """Test basic reverse rate coefficient calculation."""
    k_forward = 1e5  # m^3/(mol*s)
    dH_rxn = 50000  # J/mol
    dS_rxn = 50  # J/(mol*K)
    T = 300  # K
    
    dG_rxn = dH_rxn - T * dS_rxn
    expected = k_forward * np.exp((dG_rxn - dH_rxn) / (8.314472 * T))
    
    actual = generate_reverse_rate_coefficient(k_forward, dH_rxn, dS_rxn, T)
    
    assert np.isclose(actual, expected, rtol=1e-10)


def test_reverse_rate_with_thermo_models():
    """Test reverse rate coefficient with thermo models."""
    k_forward = 1e5  # m^3/(mol*s)
    T = 300  # K
    
    # Create mock thermo models
    # Reactants: A + B -> products
    # Let's say A has H=0, S=0 and B has H=0, S=0
    # Products: C has H=50000, S=50
    reactants = [DummyThermo(H298=0, S298=0), DummyThermo(H298=0, S298=0)]
    products = [DummyThermo(H298=50000, S298=50)]
    
    actual = generate_reverse_rate_coefficient_with_thermo(
        k_forward, reactants, products, T
    )
    
    # Expected calculation:
    # dH_rxn = 50000 - 0 = 50000 J/mol
    # dS_rxn = 50 - 0 = 50 J/(mol*K)
    # dG_rxn = 50000 - 300*50 = 50000 - 15000 = 35000 J/mol
    # k_reverse = 1e5 * exp((35000 - 50000) / (8.314472 * 300))
    dH = 50000
    dS = 50
    dG = dH - T * dS
    expected = k_forward * np.exp((dG - dH) / (8.314472 * T))
    
    assert np.isclose(actual, expected, rtol=1e-10)


def test_registry_evaluate_with_tunneling():
    """Test registry evaluation with tunneling correction."""
    forward = Arrhenius(A=1e10, n=0, Ea=50000)
    wigner = Wigner(frequency=1000)
    
    registry = RateRegistry(
        forward_model=forward,
        tunneling_model=wigner
    )
    
    T = 300
    k_no_tunneling = forward.evaluate(T)
    k_with_tunneling = registry.evaluate(T)
    
    # Should be different due to tunneling correction
    assert k_with_tunneling > k_no_tunneling
    
    # Check that the correction is applied
    wigner_factor = wigner.calculate_tunneling_factor(T)
    assert np.isclose(k_with_tunneling, k_no_tunneling * wigner_factor, rtol=1e-10)


def test_registry_reverse_rate():
    """Test registry reverse rate generation."""
    forward = Arrhenius(A=1e10, n=0, Ea=50000)
    
    registry = RateRegistry(forward_model=forward)
    
    T = 300
    
    # Mock thermo
    reactants = [DummyThermo(H298=0, S298=0), DummyThermo(H298=0, S298=0)]
    products = [DummyThermo(H298=50000, S298=50)]
    
    k_reverse = registry.reverse(reactants, products, T)
    
    # Should be positive
    assert k_reverse > 0
    
    # Verify calculation
    k_forward = forward.evaluate(T)
    dH = 50000
    dS = 50
    dG = dH - T * dS
    expected = k_forward * np.exp((dG - dH) / (8.314472 * T))
    
    assert np.isclose(k_reverse, expected, rtol=1e-10)


def test_all_model_types_evaluate():
    """Test that all model types can evaluate k(T,P)."""
    T = 300
    P = 1e5
    
    # Test each model type
    models = {
        "arrhenius": Arrhenius(A=1e10, n=0, Ea=50000),
        "lindemann": Lindemann(
            arrheniusHigh=Arrhenius(A=1e10, n=0, Ea=50000),
            arrheniusLow=Arrhenius(A=1e10, n=0, Ea=50000)
        ),
        "troe": Troe(
            arrheniusHigh=Arrhenius(A=1e10, n=0, Ea=50000),
            arrheniusLow=Arrhenius(A=1e10, n=0, Ea=50000),
            alpha=0.1,
            T1=1000,
            T2=0,
            T3=10000
        ),
        "chebyshev": Chebyshev(
            coeffs=np.array([[0, 0], [0, 0]]),
            Tmin=100,
            Tmax=5000,
            Pmin=1e3,
            Pmax=1e8
        ),
        "pdep_arrhenius": PDepArrhenius(
            A=1e10, n=0, Ea=50000,
            Pmin=1e3, Pmax=1e8
        ),
        "marcus": Marcus(
            A=1e10, n=0, Ea=50000,
            lambda_=1e6, dG=1e5
        )
    }
    
    for name, model in models.items():
        k = model.evaluate(T, P)
        assert k > 0, f"Model {name} returned non-positive rate"
        print(f"{name}: k({T}, {P}) = {k}")


def test_registry_interface():
    """Test the registry interface methods."""
    forward = Arrhenius(A=1e10, n=0, Ea=50000)
    wigner = Wigner(frequency=1000)
    
    registry = RateRegistry(
        forward_model=forward,
        tunneling_model=wigner
    )
    
    # Test forward()
    fwd = registry.forward()
    assert fwd is forward
    
    # Test evaluate()
    T = 300
    P = 1e5
    k = registry.evaluate(T, P)
    assert k > 0
    
    # Test reverse()
    reactants = [DummyThermo(H298=0, S298=0)]
    products = [DummyThermo(H298=50000, S298=50)]
    k_reverse = registry.reverse(reactants, products, T)
    assert k_reverse > 0
    
    print("Registry interface tests passed")


if __name__ == "__main__":
    # Run all tests
    test_wigner_factor_matches_rmgpy()
    test_eckart_tunneling_factor()
    test_reverse_rate_coefficient_basic()
    test_reverse_rate_with_thermo_models()
    test_registry_evaluate_with_tunneling()
    test_registry_reverse_rate()
    test_all_model_types_evaluate()
    test_registry_interface()
    print("All tests passed!")
