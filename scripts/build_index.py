# Reads the CLIP embeddings produced by embed_images.py and builds a FAISS flat
# inner-product index, then saves it to data/embeddings/faiss.index for query-time search.

import json
import sys
from pathlib import Path

import faiss
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import EMBEDDINGS_FILE, MAPPING_FILE, INDEX_FILE


def load_embeddings( embeddings_file, mapping_file ):
    """Load the embeddings array (.npy) and index-to-metadata mapping (.json) from disk."""
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    embeddings_array = np.load(str(embeddings_file)).astype("float32")
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    print(f"Loaded {len(mapping)} embeddings with shape {embeddings_array.shape}")
    return embeddings_array, mapping


def build_faiss_index( embeddings_array ):
    """Build and return a FAISS IndexFlatIP populated with all embeddings.

    IndexFlatIP (inner product) is equivalent to cosine similarity when vectors are
    L2-normalized, which is the case for all CLIP outputs produced by embed_images.py.
    """
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)
    print(f"Built FAISS index with {index.ntotal} vectors (dim={dimension})")
    return index


def save_index( index, index_file ):
    """Write the FAISS index to disk."""
    faiss.write_index(index, str(index_file))
    print(f"Saved FAISS index to {index_file}")


def load_index( index_file ):
    """Read a FAISS index from disk. Used by query.py at search time."""
    if not index_file.exists():
        raise FileNotFoundError(f"FAISS index file not found: {index_file}")
    return faiss.read_index(str(index_file))


def main():
    embeddings_array, mapping = load_embeddings(EMBEDDINGS_FILE, MAPPING_FILE)
    index = build_faiss_index(embeddings_array)
    save_index(index, INDEX_FILE)


if __name__ == "__main__":
    main()
