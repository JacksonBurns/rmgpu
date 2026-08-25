import sys
from pathlib import Path

# Add repo root to sys.path for testing
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Fixtures
EXAMPLE_DIR = Path("/home/jackson/rmgpu/RMG-Py/examples/rmg")
REF_DB = Path("/home/jackson/rmgpu/RMG-database")
RMGPY = Path("/home/jackson/rmgpu/RMG-Py/rmgpy")

def pytest_configure(config):
    pass

# Fixtures as pytest fixtures
import pytest

@pytest.fixture
def example_dir():
    return EXAMPLE_DIR

@pytest.fixture
def ref_db():
    return REF_DB

@pytest.fixture
def rmgpy():
    return RMGPY
