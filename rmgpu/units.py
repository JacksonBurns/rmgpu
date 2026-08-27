"""Quantity class backed by pint, implementing RMG's Quantity semantics."""

import re
import pint

ureg = pint.UnitRegistry()


class QuantityError(Exception):
    """Raised for invalid quantity operations or unit mismatches."""


class Quantity:
    """
    Physical quantity with units, backed by pint.

    Supports construction from (value, unit) tuples and from strings like
    "1350 K" (the legacy importer requires both).
    """

    def __init__(self, value=0.0, units=''):
        if value is None:
            value = 0.0
        self._value = float(value)
        self._units = units if units else ''

    @classmethod
    def from_string(cls, text):
        """Parse strings like '1350 K' or '-5e3 m^2/mol/s'."""
        text = text.strip()
        match = re.match(r'^([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*([a-zA-Z/\w^]+?)\s*$', text)
        if not match:
            raise QuantityError(f'Invalid quantity string: {text!r}')
        value = float(match.group(1))
        units = match.group(2).strip()
        # Convert m3 to m^3 for pint compatibility
        units = re.sub(r'\bm3\b', 'm^3', units)
        return cls(value, units)

    def to_si(self):
        """Return value in SI units."""
        if not self._units:
            return self._value
        try:
            qty = ureg.Quantity(self._value, self._units)
            return float(qty.to_base_units().magnitude)
        except Exception as e:
            raise QuantityError(f'Cannot convert {self._units} to SI: {e}')

    def value_in(self, target_units):
        """Return value in the given units."""
        try:
            qty = ureg.Quantity(self._value, self._units)
            return float(qty.to(target_units).magnitude)
        except Exception as e:
            raise QuantityError(f'Cannot convert {self._units} to {target_units}: {e}')

    def __add__(self, other):
        if not isinstance(other, Quantity):
            raise QuantityError('Cannot add non-Quantity to Quantity')
        try:
            self_qty = ureg.Quantity(self._value, self._units)
            other_qty = ureg.Quantity(other._value, other._units)
            result = self_qty + other_qty
            return Quantity(float(result.magnitude), str(result.units))
        except Exception as e:
            raise QuantityError(f'Addition failed: {e}')

    def __sub__(self, other):
        if not isinstance(other, Quantity):
            raise QuantityError('Cannot subtract non-Quantity from Quantity')
        try:
            self_qty = ureg.Quantity(self._value, self._units)
            other_qty = ureg.Quantity(other._value, other._units)
            result = self_qty - other_qty
            return Quantity(float(result.magnitude), str(result.units))
        except Exception as e:
            raise QuantityError(f'Subtraction failed: {e}')

    def __mul__(self, other):
        if not isinstance(other, Quantity):
            raise QuantityError('Cannot multiply Quantity by non-Quantity')
        try:
            result = ureg.Quantity(self._value, self._units) * ureg.Quantity(other._value, other._units)
            result = result.to_base_units()
            return Quantity(float(result.magnitude), result.units)
        except Exception as e:
            raise QuantityError(f'Multiplication failed: {e}')

    def __truediv__(self, other):
        if not isinstance(other, Quantity):
            raise QuantityError('Cannot divide Quantity by non-Quantity')
        try:
            result = ureg.Quantity(self._value, self._units) / ureg.Quantity(other._value, other._units)
            result = result.to_base_units()
            return Quantity(float(result.magnitude), result.units)
        except Exception as e:
            raise QuantityError(f'Division failed: {e}')

    def __eq__(self, other):
        if not isinstance(other, Quantity):
            return False
        try:
            self_qty = ureg.Quantity(self._value, self._units)
            other_qty = ureg.Quantity(other._value, other._units)
            if self_qty.dimensionality != other_qty.dimensionality:
                return False
            return abs(self.to_si() - other.to_si()) < 1e-12
        except Exception:
            return False

    def __repr__(self):
        return f'({self._value}, {self._units!r})'

    def __str__(self):
        return f'{self._value:g} {self._units}'

    def __radd__(self, other):
        if not isinstance(other, Quantity):
            raise QuantityError('Cannot add non-Quantity to Quantity')
        return other + self

    def __rsub__(self, other):
        if not isinstance(other, Quantity):
            raise QuantityError('Cannot subtract non-Quantity from Quantity')
        return other - self

    def __rmul__(self, other):
        if not isinstance(other, Quantity):
            raise QuantityError('Cannot multiply Quantity by non-Quantity')
        return other * self

    def __rtruediv__(self, other):
        if not isinstance(other, Quantity):
            raise QuantityError('Cannot divide Quantity by non-Quantity')
        return other / self
