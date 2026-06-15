# Loads captured images via captures.jsonl, encodes each one with the CLIP image encoder,
# and saves the resulting L2-normalized embeddings to data/embeddings/ for FAISS indexing.

import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

def read_jsonl( file ):
    """Read a JSONL file and return records as a list of dicts."""
    records = []

    try:
        with open( file, "r", encoding = "utf-8" ) as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                record = json.loads( line )
                records.append( record )

    except Exception as error:
        raise RuntimeError(f"Failed to read JSON file {file}: {error}")

    return records

    
def load_images( metadata_file ):
    """Load images listed in the metadata JSONL file. Skips missing or unreadable paths."""
    images = []

    try:
        records = read_jsonl( metadata_file )

        for record in records:
            image_path = record.get("image_path")

            if image_path is None:
                print("Warning: record is missing 'image_path'. Skipping.")
                continue

            image_path = Path( image_path)
            if not image_path.exists():
                print(f"Warning: image file does not exist: {image_path}")
                continue
            
            image = cv2.imread( str(image_path) )

            if image is None:
                print(f"Warning: failed to load image: {image_path}")
                continue

            images.append( {
                "image": image,
                "image_path": str(image_path),
                "timestamp": record.get("timestamp"),
                "file_name": record.get("file_name")
            })

    except Exception as error:
        raise RuntimeError(f"Failed to load images from metadata file {metadata_file}: {error}")

    return images

def preprocess_images( images ):
    """Convert OpenCV BGR images to RGB PIL Images as required by the CLIP processor."""
    preprocessed_images = []

    for image_record in images:
        bgr_image = image_record["image"]

        rgb_image = cv2.cvtColor( bgr_image, cv2.COLOR_BGR2RGB )
        pil_image = Image.fromarray( rgb_image )

        preprocessed_images.append( {
            "image": pil_image,
            "image_path": image_record["image_path"],
            "timestamp": image_record["timestamp"],
            "file_name": image_record["file_name"]
        })

    return preprocessed_images

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

def load_clip_model():
    """Load the CLIP processor and model from HuggingFace (~600 MB on first run)."""
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    return processor, model


def generate_embeddings( preprocessed_images, processor, model ):
    """
    Run each PIL image through the CLIP image encoder.
    Returns:
        embeddings_array: np.ndarray of shape (N, 512), L2-normalized
        mapping: list of dicts with index, image_path, timestamp, file_name
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

    embeddings_array = np.stack(embeddings, axis=0)
    return embeddings_array, mapping


def save_embeddings( base_dir, embeddings_array, mapping ):
    """Write embeddings (.npy) and index-to-metadata mapping (.json) to data/embeddings/."""
    output_dir = Path(base_dir) / "data" / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_path = output_dir / "image_embeddings.npy"
    mapping_path = output_dir / "image_mapping.json"

    np.save(str(npy_path), embeddings_array)

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print(f"Saved {len(mapping)} embeddings to {npy_path}")
    print(f"Saved mapping to {mapping_path}")

def main():
    base_dir = Path(__file__).resolve().parent.parent
    metadata_file = base_dir / "data" / "metadata" / "captures.jsonl"

    images = load_images( metadata_file )
    preprocessed_images = preprocess_images( images )
    print(f"Loaded and preprocessed {len(preprocessed_images)} images")

    print("Loading CLIP model...")
    processor, model = load_clip_model()

    print("Generating embeddings...")
    embeddings_array, mapping = generate_embeddings(preprocessed_images, processor, model)

    save_embeddings(base_dir, embeddings_array, mapping)

if __name__ == "__main__":
    main()