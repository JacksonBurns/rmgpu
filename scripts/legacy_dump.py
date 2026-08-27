#!/usr/bin/env python
"""
Test the legacy importer on all 47+ input.py files.

Runs the importer on each file and reports:
- Successfully imported
- Failed with error
- Number of import notes
"""

import os
import sys
import json
import tempfile
import shutil

# Add the repo to path
sys.path.insert(0, '/home/jackson/rmgpu/rmgpu')

from rmgpu.importer.legacy import import_legacy, LegacyImporterError


def find_input_files():
    """Find all input.py files."""
    files = []

    # From examples/rmg/
    examples_rmg = '/home/jackson/rmgpu/RMG-Py/examples/rmg'
    for dirpath, dirnames, filenames in os.walk(examples_rmg):
        if 'input.py' in filenames:
            files.append(os.path.join(dirpath, 'input.py'))

    # From test/regression/
    test_regression = '/home/jackson/rmgpu/RMG-Py/test/regression'
    if os.path.exists(test_regression):
        for dirpath, dirnames, filenames in os.walk(test_regression):
            if 'input.py' in filenames:
                files.append(os.path.join(dirpath, 'input.py'))

    return sorted(files)


def main():
    files = find_input_files()
    print(f"Found {len(files)} input.py files\n")

    results = {"success": [], "failed": [], "with_notes": []}

    for filepath in files:
        relpath = os.path.relpath(filepath, '/home/jackson/rmgpu/RMG-Py')
        try:
            result = import_legacy(filepath)

            # Check if it has import notes
            notes = result.get('import_notes', [])
            if notes:
                results["with_notes"].append({
                    "file": relpath,
                    "notes": len(notes),
                    "note_samples": notes[:3]  # First 3 notes
                })
            else:
                results["success"].append(relpath)

            # Count keys
            n_keys = len([k for k in result.keys() if k != 'import_notes'])

            print(f"  [OK] {relpath} ({n_keys} keys)")

        except LegacyImporterError as e:
            results["failed"].append({
                "file": relpath,
                "error": str(e)
            })
            print(f"  [FAIL] {relpath}: {e}")
        except Exception as e:
            results["failed"].append({
                "file": relpath,
                "error": f"{type(e).__name__}: {e}"
            })
            print(f"  [FAIL] {relpath}: {type(e).__name__}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {len(files)}")
    print(f"Success (no notes): {len(results['success'])}")
    print(f"With import notes: {len(results['with_notes'])}")
    print(f"Failed: {len(results['failed'])}")

    if results['failed']:
        print("\nFailed files:")
        for f in results['failed']:
            print(f"  - {f['file']}: {f['error']}")

    # Write results to temp file
    tmpdir = tempfile.mkdtemp(prefix="legacy_dump_")
    results_path = os.path.join(tmpdir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to: {results_path}")

    # Clean up temp dir
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()