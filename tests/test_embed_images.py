import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image


def test_read_jsonl_returns_records(tmp_path):
    from scripts.embed_images import read_jsonl
    f = tmp_path / "test.jsonl"
    f.write_text('{"a": 1}\n{"b": 2}\n')
    assert read_jsonl(f) == [{"a": 1}, {"b": 2}]


def test_read_jsonl_skips_blank_lines(tmp_path):
    from scripts.embed_images import read_jsonl
    f = tmp_path / "test.jsonl"
    f.write_text('{"a": 1}\n\n{"b": 2}\n')
    assert len(read_jsonl(f)) == 2


def test_load_images_skips_missing_paths(tmp_path):
    from scripts.embed_images import load_images
    jsonl = tmp_path / "captures.jsonl"
    jsonl.write_text(
        json.dumps({"image_path": str(tmp_path / "missing.jpg"), "timestamp": "t", "file_name": "missing.jpg"}) + "\n"
    )
    assert load_images(jsonl) == []


def test_preprocess_images_converts_bgr_to_pil(sample_bgr_image):
    from scripts.embed_images import preprocess_images
    records = [{"image": sample_bgr_image, "image_path": "x.jpg", "timestamp": "t", "file_name": "x.jpg"}]
    result = preprocess_images(records)
    assert isinstance(result[0]["image"], Image.Image)
    assert result[0]["image"].mode == "RGB"


def test_generate_embeddings_shape(mock_clip_model):
    from scripts.embed_images import generate_embeddings
    records = [
        {"image": Image.new("RGB", (64, 64)), "image_path": f"img{i}.jpg", "timestamp": "t", "file_name": f"img{i}.jpg"}
        for i in range(3)
    ]
    processor = MagicMock()
    processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
    arr, mapping = generate_embeddings(records, processor, mock_clip_model)
    assert arr.shape == (3, 512)
    assert len(mapping) == 3


def test_save_outputs_writes_both_files(tmp_path, monkeypatch):
    from scripts import embed_images
    monkeypatch.setattr(embed_images, "EMBEDDINGS_DIR",  tmp_path)
    monkeypatch.setattr(embed_images, "EMBEDDINGS_FILE", tmp_path / "image_embeddings.npy")
    monkeypatch.setattr(embed_images, "MAPPING_FILE",    tmp_path / "image_mapping.json")
    arr = np.zeros((2, 512), dtype="float32")
    mapping = [{"embedding_index": 0, "image_path": "a.jpg", "timestamp": "t", "file_name": "a.jpg"}]
    embed_images.save_embeddings(arr, mapping)
    assert (tmp_path / "image_embeddings.npy").exists()
    assert (tmp_path / "image_mapping.json").exists()


def test_incremental_embedding_skips_existing(tmp_path):
    from scripts.embed_images import get_already_embedded_paths
    mapping_file = tmp_path / "image_mapping.json"
    existing_path = str(tmp_path / "old.jpg")
    mapping_file.write_text(json.dumps([{
        "embedding_index": 0, "image_path": existing_path, "timestamp": "t", "file_name": "old.jpg"
    }]))
    already = get_already_embedded_paths(mapping_file)
    assert existing_path in already
    assert len(already) == 1
