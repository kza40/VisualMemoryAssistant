# Central configuration for the Visual Memory Assistant.
# All scripts import constants from here instead of defining them locally.

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Camera ---
CAMERA_ID = 0
CAMERA_SOURCE = "usb"       # "usb" | "csi"  (csi for Jetson GStreamer pipeline)
CAPTURE_INTERVAL_SECONDS = 10
MAX_CAPTURES = 5            # increase (or remove cap) for real deployments
IMAGE_PREFIX = "frame"

# --- CLIP ---
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# --- VLM (Moondream2) ---
VLM_MODEL_ID = "vikhyatk/moondream2"
VLM_REVISION = "2024-08-26"  # pinned for reproducibility

# --- Query ---
TOP_K = 3

# --- Paths ---
RAW_IMAGE_FOLDER = BASE_DIR / "data" / "raw"
METADATA_FOLDER  = BASE_DIR / "data" / "metadata"
METADATA_FILE    = METADATA_FOLDER / "captures.jsonl"

EMBEDDINGS_DIR  = BASE_DIR / "data" / "embeddings"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "image_embeddings.npy"
MAPPING_FILE    = EMBEDDINGS_DIR / "image_mapping.json"
INDEX_FILE      = EMBEDDINGS_DIR / "faiss.index"
