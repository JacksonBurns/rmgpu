"""Trivial test for rmgpu version import."""

def test_version_import():
    import rmgpu
    assert rmgpu.__version__ == "0.1.0"
