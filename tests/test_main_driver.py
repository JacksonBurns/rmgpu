"""Tests for main driver (job-06/step-03)."""

import pytest
from rmgpu.main import run

def test_main_driver_minimal(tmp_path):
    # Create a minimal input file
    import yaml, tempfile, os
    doc = {
        'rmgpu': '1.0',
        'species': [
            {'label': 'CH4', 'reactive': True, 'structure': {'smiles': 'C'}},
            {'label': 'O2', 'reactive': True, 'structure': {'smiles': '[O]=O'}},
        ],
        'reactors': [{
            'temperature': {'value': 1000, 'unit': 'K'},
            'pressure': {'value': 1.0, 'unit': 'bar'},
            'initial_mole_fractions': {'CH4': 1.0, 'O2': 0.0},
        }],
        'model': {
            'tolerance_move_to_core': 0.1,
            'tolerance_keep_in_edge': 0.0,
            'tolerance_interrupt_simulation': 1.0,
        }
    }
    tmp = tmp_path / 'minimal.yaml'
    with open(tmp, 'w') as f:
        yaml.dump(doc, f)

    summary = run(str(tmp))
    assert summary['core_species_count'] == 2
    assert summary['iterations'] >= 1
    # Determinism: two runs produce same labels
    summary2 = run(str(tmp))
    assert summary['core_species_labels'] == summary2['core_species_labels']
