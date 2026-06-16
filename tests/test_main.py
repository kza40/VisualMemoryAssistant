from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_cli_capture_calls_capture_main():
    with patch("scripts.capture.main") as mock:
        from main import run
        run(["capture"])
        mock.assert_called_once()


def test_cli_embed_calls_embed_main():
    with patch("scripts.embed_images.main") as mock:
        from main import run
        run(["embed"])
        mock.assert_called_once()


def test_cli_index_calls_build_index_main():
    with patch("scripts.build_index.main") as mock:
        from main import run
        run(["index"])
        mock.assert_called_once()


def test_cli_ask_requires_argument():
    from main import run
    with pytest.raises(SystemExit) as exc:
        run(["ask"])
    assert exc.value.code != 0


def test_cli_ask_end_to_end_mocked(capsys):
    with patch("scripts.query.load_resources") as mock_lr, \
         patch("scripts.query.load_clip_model") as mock_clip, \
         patch("scripts.query.embed_query") as mock_embed, \
         patch("scripts.query.search") as mock_search, \
         patch("scripts.query.load_moondream") as mock_vlm, \
         patch("scripts.query.answer_question") as mock_answer:

        mock_lr.return_value     = (MagicMock(), [])
        mock_clip.return_value   = (MagicMock(), MagicMock())
        mock_embed.return_value  = np.zeros((1, 512), dtype="float32")
        mock_search.return_value = [{
            "similarity": 0.9, "image_path": "test.jpg",
            "timestamp": "t", "file_name": "test.jpg",
        }]
        mock_vlm.return_value    = (MagicMock(), MagicMock())
        mock_answer.return_value = "The mug is on the desk."

        from main import run
        run(["ask", "where", "is", "my", "mug"])

    captured = capsys.readouterr()
    assert "The mug is on the desk." in captured.out
