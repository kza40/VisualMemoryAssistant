# Accepts a natural language query, encodes it with the CLIP text encoder, and retrieves
# the most visually similar images from the FAISS index built by build_index.py.

import json
import sys
from pathlib import Path
import numpy as np
import torch
import faiss
from transformers import CLIPModel, CLIPProcessor


BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
INDEX_FILE = EMBEDDINGS_DIR / "faiss.index"
MAPPING_FILE = EMBEDDINGS_DIR / "image_mapping.json"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
TOP_K = 3


def load_resources( index_file, mapping_file ):
    """Load the FAISS index and image mapping from disk."""
    if not index_file.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_file}. Run build_index.py first.")
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}. Run embed_images.py first.")

    index = faiss.read_index(str(index_file))

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    print(f"Loaded index with {index.ntotal} vectors")
    return index, mapping


def load_clip_model():
    """Load the CLIP processor and model for text encoding."""
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    return processor, model


def embed_query( query_text, processor, model ):
    """
    Encode a text query into a CLIP embedding.
    Returns a float32 numpy array of shape (1, 512), L2-normalized so it
    sits in the same unit-sphere space as the image embeddings.
    """
    with torch.no_grad():
        inputs = processor(text=[query_text], return_tensors="pt", padding=True)
        features = model.get_text_features(**inputs)

        # L2-normalize so inner-product search == cosine similarity
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().astype("float32")


def search( index, query_embedding, mapping, top_k ):
    """
    Search the FAISS index for the top_k most similar images.
    Returns a list of dicts: {similarity, image_path, timestamp, file_name}.
    """
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = mapping[idx]
        results.append({
            "similarity": float(score),
            "image_path": entry["image_path"],
            "timestamp": entry["timestamp"],
            "file_name": entry["file_name"],
        })

    return results


def print_results( query_text, results ):
    print(f"\nQuery: \"{query_text}\"")
    print("-" * 40)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"#{rank}  {result['file_name']}")
        print(f"    Similarity : {result['similarity']:.4f}")
        print(f"    Timestamp  : {result['timestamp']}")
        print(f"    Path       : {result['image_path']}")
        print()


def main():
    query_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter your query: ").strip()

    if not query_text:
        print("No query provided. Exiting.")
        return

    index, mapping = load_resources(INDEX_FILE, MAPPING_FILE)

    print("Loading CLIP model...")
    processor, model = load_clip_model()

    query_embedding = embed_query(query_text, processor, model)
    results = search(index, query_embedding, mapping, TOP_K)
    print_results(query_text, results)


if __name__ == "__main__":
    main()
