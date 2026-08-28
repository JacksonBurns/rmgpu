import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from rmgpu.ml.base import THERMO_CKPT_PATH, load_thermo_checkpoint
from rmgpu.molecule.molecule import Molecule
from rmgpu.ml.thermo_estimator import ThermoML, MLCoverageError

MODELS_DIR = repo_root / "models"
REFERENCES_PATH = repo_root / "gates" / "baselines" / "job04" / "reference_predictions.json"


@pytest.fixture(scope="module")
def thermo_ml():
    return ThermoML(MODELS_DIR)


@pytest.fixture(scope="module")
def reference_predictions():
    with open(REFERENCES_PATH) as f:
        return json.load(f)


def test_thermo_ml_loads_checkpoint(thermo_ml):
    assert thermo_ml.ckpt is not None
    assert thermo_ml.ckpt.checkpoint_path == THERMO_CKPT_PATH


def test_thermo_ml_covers_valid_molecules(thermo_ml):
    # Valid molecules
    assert thermo_ml.covers(Molecule("CC"))
    assert thermo_ml.covers("CC")
    assert thermo_ml.covers("CCC")

    # Uncovered due to invalid SMILES
    assert not thermo_ml.covers("invalid_smiles")
    assert not thermo_ml.covers("")

    # Uncovered due to too many heavy atoms (hypothetical)
    # We can't easily test this without a huge molecule, but the logic is there


def test_thermo_ml_predicts_valid_molecules(thermo_ml):
    # Test prediction for ethane
    prediction = thermo_ml.predict("CC")
    assert prediction.Hf298 > 0
    assert prediction.S298 > 0
    assert len(prediction.Cp_model.Cp) == 7
    assert prediction.Cp_model.Cp[0] > 0

    # Test prediction for propane
    prediction = thermo_ml.predict("CCC")
    assert prediction.Hf298 > 0
    assert prediction.S298 > 0

    # Test with Molecule object
    molecule = Molecule("CC")
    prediction = thermo_ml.predict(molecule)
    assert prediction.Hf298 > 0


def test_thermo_ml_reproduces_reference_predictions(thermo_ml, reference_predictions):
    # Get reference molecules
    ref_mols = reference_predictions["molecules"]["smiles"]
    ref_rows = reference_predictions["molecules"]["rows"]
    tol = 1e-4

    for smi in ref_mols:
        pred = thermo_ml.predict(smi)
        ref = ref_rows[smi]

        # Convert reference from log10 to actual values
        ref_Hf298 = 10 ** ref["log_H298_J_mol"]
        ref_S298 = 10 ** ref["log_S298_J_mol_K"]
        ref_Cp = [10 ** ref[f"log_Cp_{i}_J_mol_K"] for i in range(1, 8)]

        # Compare Hf298
        assert abs(pred.Hf298 - ref_Hf298) / ref_Hf298 < tol, f"Hf298 mismatch for {smi}"

        # Compare S298
        assert abs(pred.S298 - ref_S298) / ref_S298 < tol, f"S298 mismatch for {smi}"

        # Compare Cp
        for i, (pred_cp, ref_cp) in enumerate(zip(pred.Cp_model.Cp, ref_Cp)):
            assert abs(pred_cp - ref_cp) / ref_cp < tol, f"Cp_{i+1} mismatch for {smi}"


def test_thermo_ml_raises_mlcovage_error(thermo_ml):
    # Test with invalid SMILES
    with pytest.raises(MLCoverageError):
        thermo_ml.predict("invalid_smiles")


def test_thermo_ml_covers_policy(thermo_ml):
    # Test with different elements
    assert thermo_ml.covers("CC")  # Carbon
    assert thermo_ml.covers("N")   # Nitrogen
    assert thermo_ml.covers("S")   # Sulfur
    assert not thermo_ml.covers("[UuH]")  # Invalid element
