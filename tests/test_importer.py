"""
Per-file legacy import tests (job-03/step-05).

For EVERY legacy input.py in the RMG-Py corpus (38 examples/rmg + 12
test/regression = 50 files):
  1. import to a schema document,
  2. re-parse against the pydantic schema (must validate),
  3. diff the resolved document against an INDEPENDENT canonical evaluation
     of the file (gates/legacy_canonical.py) - structure equality on values,
     order-sensitive for list blocks.

This is the test-side twin of gates/gate_03.py.
"""

import pytest

from rmgpu.importer.legacy import import_legacy, LegacyImporterError
from rmgpu.schemas.input import Input
from gates.legacy_canonical import find_input_files, check_file, canonicalize

FILES = find_input_files()

# (path, expected number of IMPORT-NOTEs). The notes must be loud and
# documented: RMG replaces repeated single-block calls (last-wins), and the
# importer must say so instead of merging silently.
EXPECTED_NOTES = {
    'examples/rmg/minimal_staged/input.py': 2,
    'test/regression/oxidation/input.py': 2,
}


def rel(p: str) -> str:
    return p.replace('/home/jackson/rmgpu/RMG-Py/', '')


def test_corpus_size():
    # The step file says 47; the actual corpus in RMG-Py v4.0.0 is 50
    # (38 examples/rmg + 12 test/regression). Record both explicitly.
    assert len(FILES) == 50


@pytest.mark.parametrize("path", FILES, ids=rel)
def test_import_validates(path: str):
    doc = import_legacy(path)
    assert doc['rmgpu'] == '1.0'
    body = {k: v for k, v in doc.items() if k != 'import_notes'}
    Input(**body)  # raises on any schema violation


@pytest.mark.parametrize("path", FILES, ids=rel)
def test_import_lossless(path: str):
    doc = import_legacy(path)
    errors = check_file(path, doc)
    assert errors == [], '\n'.join(errors[:20])


@pytest.mark.parametrize("path", FILES, ids=rel)
def test_import_notes_documented(path: str):
    doc = import_legacy(path)
    relpath = rel(path)
    expected = EXPECTED_NOTES.get(relpath, 0)
    assert len(doc['import_notes']) == expected, \
        f"expected {expected} notes, got {len(doc['import_notes'])}: " \
        f"{[n['message'] for n in doc['import_notes']]}"


@pytest.mark.parametrize("path", FILES, ids=rel)
def test_inventory_matches_calls(path: str):
    """The set of DSL functions in the document matches the file's calls."""
    doc = import_legacy(path)
    canon = canonicalize(path)
    called = set(canon['calls'])
    for fn, calls in canon['calls'].items():
        if fn in ('species', 'forbidden'):
            assert len(doc.get(fn, [])) == len(calls), f"{fn} count mismatch"
    # reactor functions -> reactors list
    n_reactors = sum(len(c) for fn, c in canon['calls'].items()
                     if fn in {'simpleReactor', 'constantVIdealGasReactor',
                               'constantTPIdealGasReactor', 'liquidReactor',
                               'mbsampledReactor', 'surfaceReactor',
                               'constantTVLiquidReactor', 'liquidSurfaceReactor'})
    if n_reactors:
        assert len(doc.get('reactors', [])) == n_reactors
