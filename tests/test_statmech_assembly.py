"""
tests/test_statmech_assembly.py

Parity tests for job-07/step-03 conformer assembly.
"""

import pytest

def test_get_statmech_data_interface():
    from rmgpu.data.statmech import get_statmech_data
    from rmgpu.statmech.modes import Conformer

    # Minimal stub objects
    class DummyMol:
        def get_radical_count(self):
            return 0

    class DummyThermo:
        H298 = 0.0
        H298_unit = "J/mol"
        def get_heat_capacity(self, T):
            return 0.0

    class DummyDB:
        pass

    mol = DummyMol()
    thermo = DummyThermo()
    db = DummyDB()
    conf = get_statmech_data(mol, db, thermo)
    assert isinstance(conf, Conformer)
    assert conf.spin_multiplicity == 1
    assert conf.E0 == 0.0
    assert conf.modes == []
