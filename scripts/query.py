# Accepts a natural language query, retrieves the most visually similar images via CLIP +
# FAISS, then passes the top match to Moondream2 to generate a natural language answer.

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import INDEX_FILE, MAPPING_FILE, TOP_K
from utils.clip_utils import load_clip_model
from models.vlm import answer_question, load_moondream


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


def main( query_text=None ):
    if query_text is None:
        query_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter your query: ").strip()

    if not query_text:
        print("No query provided. Exiting.")
        return

    index, mapping = load_resources(INDEX_FILE, MAPPING_FILE)

    print("Loading CLIP model...")
    clip_processor, clip_model = load_clip_model()

    query_embedding = embed_query(query_text, clip_processor, clip_model)
    results = search(index, query_embedding, mapping, TOP_K)
    print_results(query_text, results)

    if not results:
        return

    vlm_model, tokenizer = load_moondream()
    top_image_path = results[0]["image_path"]
    print("Answer:", answer_question(top_image_path, query_text, vlm_model, tokenizer))


if __name__ == "__main__":
    main()
