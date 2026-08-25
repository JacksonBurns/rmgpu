import pytest
from rmgpu.units import Quantity, QuantityError


def test_construction_from_value_units():
    q = Quantity(100.0, 'K')
    assert q._value == 100.0
    assert q._units == 'K'

    q2 = Quantity(200.0, 'Pa')
    assert q2.value_in('bar') == pytest.approx(0.002)


def test_construction_from_string():
    q = Quantity.from_string('1350 K')
    assert q._value == 1350.0
    assert q._units == 'K'

    q2 = Quantity.from_string('5.0 m^2/mol/s')
    assert q2._value == 5.0
    assert q2._units == 'm^2/mol/s'


def test_arithmetic_addition_subtraction():
    q1 = Quantity(10.0, 'K')
    q2 = Quantity(5.0, 'K')
    assert (q1 + q2).value_in('K') == 15.0
    assert (q1 - q2).value_in('K') == 5.0

    q3 = Quantity(278.15, 'K')
    q4 = Quantity(5.0, 'delta_degC')
    assert (q3 + q4).value_in('K') == pytest.approx(283.15, rel=1e-6)


def test_arithmetic_multiplication_division():
    q1 = Quantity(2.0, 'm')
    q2 = Quantity(3.0, 'kg')
    result = q1 * q2
    assert result.to_si() == pytest.approx(6.0, rel=1e-6)

    q3 = Quantity(10.0, 'N')
    q4 = Quantity(2.0, 'm')
    result2 = q3 / q4
    assert result2.to_si() == pytest.approx(5.0, rel=1e-6)


def test_unit_mismatch_raises():
    q1 = Quantity(10.0, 'K')
    q2 = Quantity(5.0, 'm')
    with pytest.raises(QuantityError):
        _ = q1 + q2


def test_to_si_conversion():
    q = Quantity(100.0, 'K')
    assert q.to_si() == pytest.approx(100.0, rel=1e-6)

    q2 = Quantity(1.0, 'bar')
    assert q2.to_si() == pytest.approx(100000.0, rel=1e-6)


def test_string_parse_edge_cases():
    # Negative value
    q1 = Quantity.from_string('-5e3 m^2/mol/s')
    assert q1._value == -5000.0
    assert q1._units == 'm^2/mol/s'

    # Exponent
    q2 = Quantity.from_string('1e2 K')
    assert q2._value == 100.0
    assert q2._units == 'K'

    # Complex unit
    q3 = Quantity.from_string('1.5 kmol/m^3')
    assert q3._value == 1.5
    assert q3._units == 'kmol/m^3'
