"""job-04/step-01: the shared ML load path against the REAL vendored checkpoints.

Both checkpoints in models/ are loaded through rmgpu.ml.base (NOT through
models/predict.py's predictor classes - that would be circular) and predict
the fixed step-1 set (3 molecules / 2 reactions). The expected values come
from gates/baselines/job04/reference_predictions.json, which was generated
by models/predict.py ITSELF (the model team's own code, run directly), so
this round-trip is non-circular: it asserts rmgpu.ml.base reproduces the
model team's own output.
"""
import json
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE = REPO_ROOT / "gates" / "baselines" / "job04" / "reference_predictions.json"

from chemprop import data as chemprop_data  # noqa: E402
from rmgpu.ml.base import (  # noqa: E402
    KINETICS_CKPT_PATH,
    THERMO_CKPT_PATH,
    get_device,
    load_chemprop_model,
    load_kinetics_checkpoint,
    load_thermo_checkpoint,
)

TOL = 1e-4

# Tolerances for the linear-space Ea_J_mol column (gate_04.py uses the same
# pair for its round-trip check). Ea is a LINEAR target (~1e4..1e6 J/mol) in
# float32: at ~2.2e5 the float32 ULP is 2**-6 = 0.015625, and CUDA matmul
# reductions are not bit-deterministic - measured run-to-run and
# batch-size-dependent drift is up to 3 ULP (0.047 J/mol, rel 2.1e-7). A
# fixed ABSOLUTE tolerance cannot span that (this is the "test_ml_base
# flake" tracked since job-04/step-05). rtol=1e-6 covers ~8 ULP, 4.8x the
# measured worst case. The log10-space targets (thermo cols, log10_A, n)
# are stable to ~5e-7 and keep the tight absolute TOL.
EA_RTOL = 1e-6
EA_ATOL = 1e-4


def _close(actual: float, expected: float, target: str) -> bool:
    """Target-aware comparison vs the reference baseline (see EA_RTOL)."""
    if target == "Ea_J_mol":
        return abs(actual - expected) <= EA_RTOL * abs(expected) + EA_ATOL
    return abs(actual - expected) <= TOL

# The fixed set, exactly as recorded in the baseline (which holds the same
# strings the baseline script fed to the model - so test inputs == baseline
# inputs by construction).
THERMO_SMILES = ["CC", "CCC", "C[CH]CC"]  # the step file's fixed set


def _reaction_smiles():
    return _baseline()["reactions"]["smiles"]


def _baseline():
    return json.loads(BASELINE.read_text())


@pytest.fixture(scope="module")
def thermo():
    return load_thermo_checkpoint(THERMO_CKPT_PATH)


@pytest.fixture(scope="module")
def kinetics():
    return load_kinetics_checkpoint(KINETICS_CKPT_PATH)


def _mol_datapoints(smiles):
    return [
        chemprop_data.MoleculeDatapoint.from_smi(s, keep_h=True, add_h=True)
        for s in smiles
    ]


def _rxn_datapoints(smiles):
    return [
        chemprop_data.ReactionDatapoint.from_smi(s, keep_h=True, add_h=True)
        for s in smiles
    ]


# --- loading ---------------------------------------------------------------

def test_thermo_checkpoint_loads(thermo):
    assert type(thermo.model).__name__ == "MPNN"
    assert thermo.model.training is False
    from models import CHEMELEON_MOL_FEATURIZER
    assert thermo.featurizer is CHEMELEON_MOL_FEATURIZER
    assert list(thermo.targets) == [
        "log_H298_J_mol",
        "log_S298_J_mol_K",
        "log_Cp_1_J_mol_K", "log_Cp_2_J_mol_K", "log_Cp_3_J_mol_K",
        "log_Cp_4_J_mol_K", "log_Cp_5_J_mol_K", "log_Cp_6_J_mol_K",
        "log_Cp_7_J_mol_K",
    ]


def test_kinetics_checkpoint_loads(kinetics):
    assert type(kinetics.model).__name__ == "MPNN"
    assert kinetics.model.training is False
    from models import RIGR_RXN_FEATURIZER
    assert kinetics.featurizer is RIGR_RXN_FEATURIZER
    assert list(kinetics.targets) == ["log10_A", "n", "Ea_J_mol"]


def test_unknown_checkpoint_rejected():
    with pytest.raises(ValueError, match="unknown checkpoint"):
        load_chemprop_model(REPO_ROOT / "gates" / "baselines" / "job04" / "reference_predictions.json")


# --- the round-trip vs the model team's own output --------------------------

def test_thermo_matches_reference(thermo):
    ref = _baseline()["molecules"]
    assert ref["smiles"] == THERMO_SMILES
    preds = thermo.predict_raw(_mol_datapoints(THERMO_SMILES))
    assert isinstance(preds, torch.Tensor)
    assert preds.shape == (3, 9)
    for i, smi in enumerate(THERMO_SMILES):
        for j, target in enumerate(thermo.targets):
            expected = ref["rows"][smi][target]
            actual = float(preds[i, j])
            assert abs(actual - expected) <= TOL, (
                f"{smi} {target}: rmgpu {actual!r} vs reference {expected!r}"
            )


def test_kinetics_matches_reference(kinetics):
    ref = _baseline()["reactions"]
    smiles = _reaction_smiles()
    assert ref["smiles"] == smiles
    preds = kinetics.predict_raw(_rxn_datapoints(smiles))
    assert isinstance(preds, torch.Tensor)
    assert preds.shape == (2, 3)
    for i, smi in enumerate(smiles):
        for j, target in enumerate(kinetics.targets):
            expected = ref["rows"][smi][target]
            actual = float(preds[i, j])
            assert _close(actual, expected, target), (
                f"{smi[:40]}... {target}: rmgpu {actual!r} vs reference {expected!r}"
            )


def test_thermo_reference_sanity():
    """The baseline itself is physically sensible (log10-space): 10^pred
    lands on the scale the models were trained on (H298 ~1e5 J/mol =
    O(100) kJ/mol; S298 ~100s J/mol/K; Cp 7-point grid ~20-500 J/mol/K),
    and the kinetics Ea are real barriers (1e4..1e6 J/mol)."""
    ref = _baseline()["molecules"]["rows"]["CC"]
    h298 = 10 ** ref["log_H298_J_mol"]
    assert 1e4 < h298 < 1e6  # J/mol, O(100) kJ/mol
    s298 = 10 ** ref["log_S298_J_mol_K"]
    assert 50.0 < s298 < 500.0  # J/mol/K
    cp = [10 ** ref[f"log_Cp_{i}_J_mol_K"] for i in range(1, 8)]
    assert all(20.0 < c < 500.0 for c in cp)  # J/mol/K
    # Ea from the kinetics reference is a real barrier: 1e4..1e6 J/mol.
    for row in _baseline()["reactions"]["rows"].values():
        assert 1e4 < row["Ea_J_mol"] < 1e6


# --- determinism ------------------------------------------------------------
# CUDA kernels are not bit-deterministic (reduction order). The log10 columns
# are stable to ~5e-7; the linear Ea column drifts a few float32 ULP between
# consecutive runs (see EA_RTOL / _assert_same). On CPU the runs are
# bit-identical and the same assertions hold.

def _assert_same(p1, p2, what, targets):
    # Per-column tolerances: the linear Ea column is not bit-stable on CUDA
    # (a few float32 ULP between consecutive runs, see EA_RTOL); the log10
    # columns are stable to ~5e-7.
    for j, target in enumerate(targets):
        d = float((p1[:, j] - p2[:, j]).abs().max())
        if target == "Ea_J_mol":
            assert d <= EA_RTOL * float(p1[:, j].abs().max()) + EA_ATOL, (
                f"{what} {target}: two runs differ by {d}"
            )
        else:
            assert d <= 1e-6, f"{what} {target}: two runs differ by {d}"


def test_thermo_deterministic(thermo):
    p1 = thermo.predict_raw(_mol_datapoints(THERMO_SMILES))
    p2 = thermo.predict_raw(_mol_datapoints(THERMO_SMILES))
    _assert_same(p1, p2, "thermo", thermo.targets)


def test_kinetics_deterministic(kinetics):
    smiles = _reaction_smiles()
    p1 = kinetics.predict_raw(_rxn_datapoints(smiles))
    p2 = kinetics.predict_raw(_rxn_datapoints(smiles))
    _assert_same(p1, p2, "kinetics", kinetics.targets)


# --- plumbing ---------------------------------------------------------------

def test_get_device_returns_valid_device():
    dev = get_device()
    assert dev.type in ("cpu", "cuda")
    if dev.type == "cuda":
        assert torch.cuda.is_available()


def test_single_item_prediction_not_dropped(thermo, kinetics):
    """Regression for chemprop's auto drop_last (last batch of size 1 is
    dropped by default) - predict_raw must return one row per input even
    for a single input (len % batch_size == 1)."""
    p = thermo.predict_raw(_mol_datapoints(["CC"]))
    assert p.shape == (1, 9)
    ref = _baseline()["molecules"]["rows"]["CC"]
    for j, target in enumerate(thermo.targets):
        assert abs(float(p[0, j]) - ref[target]) <= TOL
    smiles = _reaction_smiles()
    q = kinetics.predict_raw(_rxn_datapoints(smiles[:1]))
    assert q.shape == (1, 3)
    ref0 = _baseline()["reactions"]["rows"][smiles[0]]
    for j, target in enumerate(kinetics.targets):
        assert _close(float(q[0, j]), ref0[target], target)
