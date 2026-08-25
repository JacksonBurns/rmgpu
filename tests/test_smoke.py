"""Smoke tests: imports, version, CUDA availability."""
import subprocess
import sys

def test_version_import():
    import rmgpu
    assert rmgpu.__version__ == "0.1.0"

def test_subpackage_imports():
    subpackages = [
        "core", "molecule", "db", "ml", "kinetics", "pdep", "statmech",
        "reactor", "io", "schemas", "tools", "sensitivity", "plugins"
    ]
    for sp in subpackages:
        __import__(f"rmgpu.{sp}")

def test_rmgpu_version_command():
    result = subprocess.run(
        [sys.executable, "-m", "rmgpu.version"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "0.1.0"

def test_required_imports():
    import rdkit
    import torch
    import chemprop
    import cantera
    import chemicals
    import fluids
    import thermo
    import pint
    import sqlalchemy
    import torchdae
    import pydantic

def test_cuda_available():
    import torch
    assert torch.cuda.is_available()
