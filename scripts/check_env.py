#!/usr/bin/env python3
"""Verify that every dependency required by the rmgpu job set imports cleanly."""
import importlib.metadata
import importlib.util
import os
import sys

EXPECTED_PKGS = [
    "numpy", "scipy", "rdkit", "pint", "chemprop", "cantera", "chemicals",
    "fluids", "thermo", "torchdae", "sqlalchemy", "polars", "pydantic",
    "click", "pytest",
]
DB_PATHS = [
    "/home/jackson/rmgpu/rmgdb/db/thermo.db",
    "/home/jackson/rmgpu/rmgdb/db/kinetics.db",
    "/home/jackson/rmgpu/rmgdb/db/transport.db",
    "/home/jackson/rmgpu/rmgdb/db/solvation.db",
    "/home/jackson/rmgpu/rmgdb/db/statmech.db",
]
CHECKPOINT_PATH = "/home/jackson/rmgpu/chemprop_example/example_model_v2_regression_mol.ckpt"

failed = False
for pkg in EXPECTED_PKGS:
    try:
        version = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        spec = importlib.util.find_spec(pkg)
        if spec is None:
            print(f"MISSING: {pkg}")
            failed = True
            continue
        version = "no installed version metadata"
    print(f"{pkg}: {version}")
    __import__(pkg)

import torch
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("WARNING: torch CUDA unavailable; install a CUDA build if this environment will run GPU work.")
    failed = True

try:
    import rmgdb
    print("rmgdb: importable")
except ImportError as exc:
    print(f"MISSING rmgdb: {exc}")
    failed = True

for path in DB_PATHS:
    print(f"DB {'present' if os.path.exists(path) else 'MISSING'}: {path}")
    if not os.path.exists(path):
        failed = True

checkpoint_present = os.path.exists(CHECKPOINT_PATH)
print(f"checkpoint {'present' if checkpoint_present else 'MISSING'}: {CHECKPOINT_PATH}")
if not checkpoint_present:
    failed = True

sys.exit(1 if failed else 0)
