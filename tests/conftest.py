import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_sample_data  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def sample_env(tmp_path_factory):
    """Point every module at a small synthetic dataset so tests never touch real data."""
    base = tmp_path_factory.mktemp("ecom")
    raw = base / "raw"
    make_sample_data.write(raw, n_orders=4000, seed=7)
    mp = pytest.MonkeyPatch()
    mp.setenv("ECOM_DATA_DIR", str(raw))
    mp.setenv("ECOM_PROCESSED_DIR", str(base / "processed"))
    mp.setenv("ECOM_MODELS_DIR", str(base / "models"))
    yield base
    mp.undo()


@pytest.fixture(scope="session")
def tables(sample_env):
    from ecom.data.load import load_all

    return load_all()
