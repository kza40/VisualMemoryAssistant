import json
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


def test_open_camera_csi_uses_gstreamer(monkeypatch):
    from scripts import capture
    with patch("cv2.VideoCapture") as mock_cap:
        mock_instance = mock_cap.return_value
        mock_instance.isOpened.return_value = True
        capture.open_camera(0, camera_source="csi")
        args = mock_cap.call_args[0]
        assert "nvarguscamerasrc" in args[0]


def test_open_camera_usb_uses_integer_id(monkeypatch):
    from scripts import capture
    with patch("cv2.VideoCapture") as mock_cap:
        mock_instance = mock_cap.return_value
        mock_instance.isOpened.return_value = True
        capture.open_camera(0, camera_source="usb")
        args = mock_cap.call_args[0]
        assert args[0] == 0


def test_create_directories_creates_folders(tmp_path, monkeypatch):
    from scripts import capture
    monkeypatch.setattr(capture, "RAW_IMAGE_FOLDER", tmp_path / "raw")
    monkeypatch.setattr(capture, "METADATA_FOLDER",  tmp_path / "metadata")
    capture.create_directories()
    assert (tmp_path / "raw").exists()
    assert (tmp_path / "metadata").exists()


def test_generate_timestamps_format():
    from scripts.capture import generate_timestamps
    filename_ts, iso_ts = generate_timestamps()
    assert re.match(r"\d{8}_\d{6}", filename_ts)
    datetime.fromisoformat(iso_ts)  # raises ValueError if unparseable


def test_save_frame_writes_file(tmp_path, monkeypatch):
    from scripts import capture
    monkeypatch.setattr(capture, "RAW_IMAGE_FOLDER", tmp_path)
    with patch("cv2.imwrite", return_value=True) as mock_write:
        path = capture.save_frame(np.zeros((64, 64, 3), dtype=np.uint8), "20260614_120000")
    mock_write.assert_called_once()
    assert path.name == "frame_20260614_120000.jpg"


def test_save_frame_raises_on_failure(tmp_path, monkeypatch):
    from scripts import capture
    monkeypatch.setattr(capture, "RAW_IMAGE_FOLDER", tmp_path)
    with patch("cv2.imwrite", return_value=False):
        with pytest.raises(RuntimeError):
            capture.save_frame(np.zeros((64, 64, 3), dtype=np.uint8), "20260614_120000")


def test_append_metadata_writes_valid_json(tmp_path, monkeypatch):
    from scripts import capture
    metadata_file = tmp_path / "captures.jsonl"
    monkeypatch.setattr(capture, "METADATA_FILE", metadata_file)
    capture.append_metadata(tmp_path / "frame_test.jpg", "2026-06-14T12:00:00")
    record = json.loads(metadata_file.read_text())
    assert record["timestamp"] == "2026-06-14T12:00:00"
    assert record["file_name"] == "frame_test.jpg"
    assert record["status"] == "captured"
    assert "image_path" in record


def test_append_metadata_appends_not_overwrites(tmp_path, monkeypatch):
    from scripts import capture
    metadata_file = tmp_path / "captures.jsonl"
    monkeypatch.setattr(capture, "METADATA_FILE", metadata_file)
    image_path = tmp_path / "frame_test.jpg"
    capture.append_metadata(image_path, "2026-06-14T12:00:00")
    capture.append_metadata(image_path, "2026-06-14T12:00:10")
    lines = [l for l in metadata_file.read_text().strip().split("\n") if l]
    assert len(lines) == 2
