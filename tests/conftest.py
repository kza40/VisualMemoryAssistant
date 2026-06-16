# Shared pytest fixtures for the Visual Memory Assistant test suite.
# All fixtures are designed to avoid real hardware, model downloads, or disk I/O
# outside of pytest's tmp_path — so the suite runs fast and offline.

import sys
from pathlib import Path
from unittest.mock import MagicMock

import faiss
import numpy as np
import pytest
import torch

# Ensure the repo root is importable so `from scripts.x import ...` works in every test
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temp directory tree mirroring data/."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "metadata").mkdir()
    (tmp_path / "embeddings").mkdir()
    return tmp_path


@pytest.fixture
def sample_bgr_image():
    """Small synthetic BGR image as a NumPy array (no file I/O)."""
    return np.zeros((64, 64, 3), dtype=np.uint8)


@pytest.fixture
def mock_clip_model():
    """Returns a fixed (1, 512) tensor on any get_image/text_features call."""
    model = MagicMock()
    model.get_image_features.return_value = torch.ones(1, 512)
    model.get_text_features.return_value = torch.ones(1, 512)
    return model


@pytest.fixture
def mock_faiss_index():
    """Tiny in-memory IndexFlatIP with 3 pre-inserted random L2-normalized vectors."""
    dim = 512
    index = faiss.IndexFlatIP(dim)
    vecs = np.random.randn(3, dim).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index.add(vecs)
    return index
