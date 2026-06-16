import json
from unittest.mock import MagicMock

import faiss
import numpy as np
import pytest
import torch


def _mapping(n=3):
    return [
        {"embedding_index": i, "image_path": f"img{i}.jpg",
         "timestamp": f"2026-06-14T12:00:0{i}", "file_name": f"img{i}.jpg"}
        for i in range(n)
    ]


def _unit_query():
    q = np.random.randn(1, 512).astype("float32")
    q /= np.linalg.norm(q)
    return q


def test_embed_text_query_shape(mock_clip_model):
    from scripts.query import embed_query
    processor = MagicMock()
    processor.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
    result = embed_query("test query", processor, mock_clip_model)
    assert result.shape == (1, 512)


def test_embed_text_query_is_normalized(mock_clip_model):
    from scripts.query import embed_query
    processor = MagicMock()
    processor.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
    result = embed_query("test query", processor, mock_clip_model)
    assert abs(np.linalg.norm(result) - 1.0) < 1e-5


def test_load_resources_returns_index_and_mapping(tmp_path):
    from scripts.query import load_resources
    vecs = np.random.randn(3, 512).astype("float32")
    index = faiss.IndexFlatIP(512)
    index.add(vecs)
    index_f   = tmp_path / "faiss.index"
    mapping_f = tmp_path / "image_mapping.json"
    faiss.write_index(index, str(index_f))
    mapping_f.write_text(json.dumps(_mapping()))
    idx, mapping = load_resources(index_f, mapping_f)
    assert idx.ntotal == 3
    assert len(mapping) == 3


def test_search_returns_correct_count(mock_faiss_index):
    from scripts.query import search
    results = search(mock_faiss_index, _unit_query(), _mapping(), top_k=2)
    assert len(results) == 2


def test_search_results_have_required_keys(mock_faiss_index):
    from scripts.query import search
    results = search(mock_faiss_index, _unit_query(), _mapping(), top_k=1)
    assert {"similarity", "image_path", "timestamp", "file_name"} <= results[0].keys()


def test_search_score_order_descending(mock_faiss_index):
    from scripts.query import search
    results = search(mock_faiss_index, _unit_query(), _mapping(), top_k=3)
    scores = [r["similarity"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_with_empty_index_returns_empty():
    from scripts.query import search
    empty_index = faiss.IndexFlatIP(512)
    results = search(empty_index, np.zeros((1, 512), dtype="float32"), [], top_k=3)
    assert results == []
