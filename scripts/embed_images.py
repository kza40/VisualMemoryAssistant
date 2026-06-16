# Loads captured images via captures.jsonl, encodes each one with the CLIP image encoder,
# and saves the resulting L2-normalized embeddings to data/embeddings/ for FAISS indexing.
# Supports incremental embedding — already-embedded images are skipped on subsequent runs.

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import METADATA_FILE, EMBEDDINGS_DIR, EMBEDDINGS_FILE, MAPPING_FILE
from utils.clip_utils import load_clip_model


def read_jsonl( file ):
    """Read a JSONL file and return records as a list of dicts."""
    records = []
    try:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    except Exception as error:
        raise RuntimeError(f"Failed to read JSON file {file}: {error}")
    return records


def load_images( metadata_file ):
    """Load images listed in the metadata JSONL file. Skips missing or unreadable paths."""
    images = []
    try:
        for record in read_jsonl(metadata_file):
            image_path = record.get("image_path")
            if image_path is None:
                print("Warning: record is missing 'image_path'. Skipping.")
                continue
            image_path = Path(image_path)
            if not image_path.exists():
                print(f"Warning: image file does not exist: {image_path}")
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: failed to load image: {image_path}")
                continue
            images.append({
                "image": image,
                "image_path": str(image_path),
                "timestamp": record.get("timestamp"),
                "file_name": record.get("file_name"),
            })
    except Exception as error:
        raise RuntimeError(f"Failed to load images from {metadata_file}: {error}")
    return images


def preprocess_images( images ):
    """Convert OpenCV BGR images to RGB PIL Images as required by the CLIP processor."""
    preprocessed = []
    for record in images:
        bgr = record["image"]
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        preprocessed.append({
            "image": pil,
            "image_path": record["image_path"],
            "timestamp": record["timestamp"],
            "file_name": record["file_name"],
        })
    return preprocessed


def generate_embeddings( preprocessed_images, processor, model ):
    """
    Run each PIL image through the CLIP image encoder.
    Returns (embeddings_array, mapping) where embeddings_array is shape (N, 512),
    L2-normalized, and mapping is a list of index-to-metadata dicts.
    """
    embeddings = []
    mapping = []
    with torch.no_grad():
        for idx, record in enumerate(preprocessed_images):
            inputs = processor(images=record["image"], return_tensors="pt")
            features = model.get_image_features(**inputs)

            # L2-normalize so that inner-product search equals cosine similarity
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.squeeze(0).cpu().numpy())

            mapping.append({
                "embedding_index": idx,
                "image_path": record["image_path"],
                "timestamp": record["timestamp"],
                "file_name": record["file_name"],
            })
    return np.stack(embeddings, axis=0), mapping


def get_already_embedded_paths( mapping_file ):
    """Return the set of image paths already present in the embedding mapping."""
    if not Path(mapping_file).exists():
        return set()
    with open(mapping_file, "r", encoding="utf-8") as f:
        return {entry["image_path"] for entry in json.load(f)}


def load_existing_embeddings( embeddings_file, mapping_file ):
    """Load existing embeddings array and mapping if both files exist, else return empty."""
    if not Path(embeddings_file).exists() or not Path(mapping_file).exists():
        return None, []
    arr = np.load(str(embeddings_file)).astype("float32")
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return arr, mapping


def save_embeddings( embeddings_array, mapping ):
    """Write embeddings (.npy) and index-to-metadata mapping (.json) to data/embeddings/."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(EMBEDDINGS_FILE), embeddings_array)
    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved {len(mapping)} embeddings to {EMBEDDINGS_FILE}")
    print(f"Saved mapping to {MAPPING_FILE}")


def main():
    already_embedded = get_already_embedded_paths(MAPPING_FILE)
    all_images = load_images(METADATA_FILE)
    new_images = [img for img in all_images if img["image_path"] not in already_embedded]

    if not new_images:
        print(f"No new images to embed. ({len(already_embedded)} already embedded)")
        return

    print(f"Embedding {len(new_images)} new images ({len(already_embedded)} already embedded)...")
    preprocessed = preprocess_images(new_images)

    print("Loading CLIP model...")
    processor, model = load_clip_model()

    new_embeddings, new_mapping = generate_embeddings(preprocessed, processor, model)

    existing_embeddings, existing_mapping = load_existing_embeddings(EMBEDDINGS_FILE, MAPPING_FILE)

    if existing_embeddings is not None:
        # Re-index new entries to continue from where the existing mapping left off
        offset = len(existing_mapping)
        for entry in new_mapping:
            entry["embedding_index"] += offset
        combined_embeddings = np.vstack([existing_embeddings, new_embeddings])
        combined_mapping = existing_mapping + new_mapping
    else:
        combined_embeddings = new_embeddings
        combined_mapping = new_mapping

    save_embeddings(combined_embeddings, combined_mapping)


if __name__ == "__main__":
    main()
