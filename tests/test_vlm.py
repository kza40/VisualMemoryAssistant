from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


def test_load_moondream_uses_cpu_when_no_cuda():
    from models.vlm import load_moondream
    with patch("torch.cuda.is_available", return_value=False), \
         patch("models.vlm.AutoTokenizer.from_pretrained", return_value=MagicMock()), \
         patch("models.vlm.AutoModelForCausalLM.from_pretrained") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_instance.to.return_value = mock_instance
        load_moondream()
        mock_instance.to.assert_called_with("cpu")


def test_answer_question_returns_string(tmp_path):
    from models.vlm import answer_question
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64)).save(str(img_path))
    model = MagicMock()
    model.encode_image.return_value = MagicMock()
    model.answer_question.return_value = "There is a desk."
    result = answer_question(str(img_path), "What do you see?", model, MagicMock())
    assert isinstance(result, str)
    assert len(result) > 0


def test_answer_question_raises_on_missing_image():
    from models.vlm import answer_question
    with pytest.raises(FileNotFoundError):
        answer_question("/nonexistent/path.jpg", "What is this?", MagicMock(), MagicMock())
