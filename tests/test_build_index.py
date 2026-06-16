import json

import faiss
import numpy as np
import pytest


def _random_vecs(n=3, dim=512):
    vecs = np.random.randn(n, dim).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def test_build_index_creates_faiss_file(tmp_path, monkeypatch):
    from scripts import build_index
    npy       = tmp_path / "image_embeddings.npy"
    mapping_f = tmp_path / "image_mapping.json"
    index_f   = tmp_path / "faiss.index"
    np.save(str(npy), _random_vecs())
    mapping_f.write_text(json.dumps([{"embedding_index": i} for i in range(3)]))
    monkeypatch.setattr(build_index, "EMBEDDINGS_FILE", npy)
    monkeypatch.setattr(build_index, "MAPPING_FILE",    mapping_f)
    monkeypatch.setattr(build_index, "INDEX_FILE",      index_f)
    build_index.main()
    assert index_f.exists()


def test_index_dimension_matches_embeddings():
    from scripts.build_index import build_faiss_index
    index = build_faiss_index(_random_vecs())
    assert index.d == 512


def test_index_search_returns_top_k(mock_faiss_index):
    query = np.random.randn(1, 512).astype("float32")
    query /= np.linalg.norm(query)
    _, indices = mock_faiss_index.search(query, 2)
    assert len(indices[0]) == 2


def test_build_index_raises_on_missing_npy(tmp_path, monkeypatch):
    from scripts import build_index
    monkeypatch.setattr(build_index, "EMBEDDINGS_FILE", tmp_path / "missing.npy")
    monkeypatch.setattr(build_index, "MAPPING_FILE",    tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        build_index.main()
