# Shared CLIP model loader used by both embed_images.py and query.py.
# Centralised here so the model ID is never duplicated across scripts.

from transformers import CLIPModel, CLIPProcessor
from config import CLIP_MODEL_ID


def load_clip_model():
    """Load the CLIP processor and model from HuggingFace (~600 MB on first run)."""
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    return processor, model
