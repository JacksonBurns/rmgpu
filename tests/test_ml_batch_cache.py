import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from rmgpu.ml.kinetics_estimator import KineticsML

def test_cache_hit_avoids_second_call():
    # Create a mock checkpoint
    mock_ckpt = MagicMock()
    # Return fake raw predictions
    raw_tensor = MagicMock()
    raw_tensor.detach.return_value.cpu.return_value.numpy.return_value = [[0.0, 0.0, 0.0]]
    mock_ckpt.predict_raw = MagicMock(return_value=[raw_tensor])

    # Patch load_kinetics_checkpoint to return mock
    with patch('rmgpu.ml.kinetics_estimator.load_kinetics_checkpoint', return_value=mock_ckpt):
        with patch('rmgpu.ml.kinetics_estimator.ThermoML'):
            ml = KineticsML()
            # Mock covers to always True
            ml.covers = lambda reaction_smiles: True
            # Predict twice
            p1 = ml.predict("C>>C", degeneracy=1.0)
            p2 = ml.predict("C>>C", degeneracy=1.0)
            # Second call should be from cache, so predict_raw called once
            assert mock_ckpt.predict_raw.call_count == 1
            assert p1.A == p2.A
            assert p1.n == p2.n
            assert p1.Ea == p2.Ea
