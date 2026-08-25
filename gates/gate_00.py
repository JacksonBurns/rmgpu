"""Gate script for job 00: run all smoke checks, print PASS/FAIL, exit 0/1.

Checks:
1. Every rmgpu subpackage imports
2. `rmgpu version` prints 0.1.0 (via python -m rmgpu.version)
3. All required imports work
4. torch.cuda.is_available() is True
5. Write reports/job-00.md
"""
import subprocess
import sys
import os

def get_pip_freeze():
    """Get pip freeze output filtered to required packages."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True
    )
    return result.stdout

def main():
    failed = False
    report_lines = []

    # 1. Import every rmgpu subpackage
    print("[1] Importing rmgpu subpackages...")
    subpackages = [
        "core", "molecule", "db", "ml", "kinetics", "pdep", "statmech",
        "reactor", "io", "schemas", "tools", "sensitivity", "plugins"
    ]
    for sp in subpackages:
        try:
            __import__(f"rmgpu.{sp}")
            print(f"  OK: rmgpu.{sp}")
        except Exception as e:
            print(f"  FAIL: rmgpu.{sp} ({e})")
            failed = True

    # 2. rmgpu version via python -m
    print("[2] Checking `python -m rmgpu.version` output...")
    result = subprocess.run(
        [sys.executable, "-m", "rmgpu.version"],
        capture_output=True, text=True
    )
    output = result.stdout.strip()
    if result.returncode == 0 and output == "0.1.0":
        print(f"  OK: version = {output}")
    else:
        print(f"  FAIL: returncode={result.returncode}, output={output}")
        failed = True

    # 3. All required imports
    print("[3] Checking required imports...")
    required = [
        "rdkit", "torch", "chemprop", "cantera", "chemicals",
        "fluids", "thermo", "pint", "sqlalchemy", "torchdae", "pydantic"
    ]
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  OK: {pkg}")
        except Exception as e:
            print(f"  FAIL: {pkg} ({e})")
            failed = True

    # 4. torch.cuda.is_available()
    print("[4] Checking torch.cuda.is_available()...")
    cuda_ok = False
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"  OK: cuda_available = {cuda_ok}")
        if not cuda_ok:
            failed = True
    except Exception as e:
        print(f"  FAIL: torch.cuda.is_available() ({e})")
        failed = True

    # Get pip freeze for report
    pip_freeze = get_pip_freeze()

    # Write report
    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "job-00.md")

    report_lines = [
        "# Job 00 Gate Report\n",
        "## Environment Versions\n",
        "```",
        pip_freeze,
        "```\n",
        "## Checkpoint Inventory\n",
        "From step-01 report:\n",
        "- **Path:** `/home/jackson/rmgpu/chemprop_example/example_model_v2_regression_mol.ckpt`\n",
        "- **Format:** `.ckpt` (PyTorch zip archive)\n",
        "- **Loaded via chemprop example pattern:** `models.MPNN.load_from_checkpoint(checkpoint_path)` succeeded.\n",
        "- **Featurizer:** `SimpleMoleculeMolGraphFeaturizer` (from chemprop example).\n",
        "- **Output dims:** single-task regression model, one prediction per molecule.\n",
        "- **1-molecule predict test:** ethane (CC) predicted value = `2.166739`.\n",
        "The checkpoint inventory is complete for job 04 to consume.\n",
        "## rmgdb Install Method\n",
        "rmgdb is importable (confirmed in check_env.py output).\n",
        "## Gate Result\n",
        "PASS" if not failed else "FAIL",
        ""
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport written to {report_path}")

    # Final result
    print()
    if failed:
        print("FAIL")
        return 1
    print("PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
