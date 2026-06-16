# Wrapper for the Moondream2 vision-language model — loads the model and answers
# a natural language question about a given image (~2 GB download on first run).

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import VLM_MODEL_ID, VLM_REVISION


def load_moondream():
    """
    Load the Moondream2 model and tokenizer.
    Automatically uses CUDA if available, otherwise falls back to CPU.
    Returns (model, tokenizer).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Moondream2 on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(VLM_MODEL_ID, revision=VLM_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        VLM_MODEL_ID,
        revision=VLM_REVISION,
        trust_remote_code=True,
        # fp16 halves VRAM usage on GPU; fp32 required on CPU
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    return model, tokenizer


def answer_question( image_path, question, model, tokenizer ):
    """Open image_path and ask Moondream2 the given question. Returns the answer string."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    enc_image = model.encode_image(image)
    return model.answer_question(enc_image, question, tokenizer)
