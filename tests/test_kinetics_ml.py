import pytest
from pathlib import Path

from rmgpu.ml.kinetics_estimator import KineticsML, KineticsPrediction, MLCoverageError

# Get repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
REFERENCE_FILE = REPO_ROOT / "gates" / "baselines" / "job04" / "reference_predictions.json"

@pytest.fixture(scope="module")
def kinetics_ml():
    return KineticsML(MODELS_DIR)

@pytest.fixture(scope="module")
def reference_predictions():
    import json
    with open(REFERENCE_FILE) as f:
        return json.load(f)

def test_loads_checkpoint(kinetics_ml):
    assert kinetics_ml.ckpt is not None
    assert kinetics_ml.ckpt.featurizer is not None
    assert kinetics_ml.ckpt.targets == ("log10_A", "n", "Ea_J_mol")

def test_predict_reproduces_reference_predictions(kinetics_ml, reference_predictions):
    reactions = reference_predictions["reactions"]
    rxn_smiles_list = reactions["smiles"]
    
    for rxn_smiles in rxn_smiles_list:
        pred = kinetics_ml.predict(rxn_smiles, degeneracy=1.0)
        
        # Reference values are log10_A, n, Ea_J_mol
        ref_log10_A = reactions["rows"][rxn_smiles]["log10_A"]
        ref_n = reactions["rows"][rxn_smiles]["n"]
        ref_Ea = reactions["rows"][rxn_smiles]["Ea_J_mol"]
        
        # Convert reference to A for comparison
        ref_A = 10 ** ref_log10_A
        
        # Check with tolerance (relative for n and Ea)
        assert abs(pred.A - ref_A) / ref_A < 1e-4, f"A mismatch: {pred.A} vs {ref_A}"
        assert abs(pred.n - ref_n) / max(ref_n, 1e-10) < 1e-4, f"n mismatch: {pred.n} vs {ref_n}"
        assert abs(pred.Ea - ref_Ea) / max(ref_Ea, 1e-10) < 1e-4, f"Ea mismatch: {pred.Ea} vs {ref_Ea}"

def test_degeneracy_conversion(kinetics_ml, reference_predictions):
    """Verify A*degeneracy boundary conversion."""
    reactions = reference_predictions["reactions"]
    rxn_smiles = reactions["smiles"][0]
    
    # Predict with degeneracy=1
    pred_1 = kinetics_ml.predict(rxn_smiles, degeneracy=1.0)
    A_1 = pred_1.A
    
    # Predict with degeneracy=2
    pred_2 = kinetics_ml.predict(rxn_smiles, degeneracy=2.0)
    A_2 = pred_2.A
    
    # A should double when degeneracy doubles
    assert abs(A_2 - 2 * A_1) / A_1 < 1e-4, f"Degeneracy conversion failed: {A_2} vs {2 * A_1}"

def test_coverage_cases(kinetics_ml):
    """Test coverage raises MLCoverageError for uncovered reactions."""
    # Valid reaction from reference
    valid_rxn = "[O:1]([C:2]([C:3]([C:4](=[O:5])[C:6]([O:7][H:15])([H:13])[H:14])([H:11])[H:12])([H:9])[H:10])[H:8]>>[C:3](=[C:4]=[O:5])([H:11])[H:12].[C:6]([O:7][H:15])([H:8])([H:13])[H:14].[O:1]=[C:2]([H:9])[H:10]"
    assert kinetics_ml.covers(valid_rxn) is True
    
    # Invalid reaction (no >>)
    assert kinetics_ml.covers("CC") is False
    
    # Invalid reaction (malformed)
    assert kinetics_ml.covers("C>>") is False
    
    # Try to predict with invalid reaction
    with pytest.raises(MLCoverageError):
        kinetics_ml.predict("CC")
    
    with pytest.raises(MLCoverageError):
        kinetics_ml.predict("C>>")