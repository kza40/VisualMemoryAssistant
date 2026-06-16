# Visual Memory Assistant

A local AI-powered visual memory system that periodically captures images from a camera, stores them as searchable memories, and lets the user ask natural language questions like:

> "Where did I last see my coffee mug?"

The system uses computer vision, multimodal embeddings, a vector database, and a Vision-Language Model to retrieve relevant past images and explain what was seen.

---

## Overview

The Visual Memory Assistant is designed to give a device a searchable memory of its environment.

The MVP runs on a local machine or Jetson device with a connected webcam. It periodically captures images, embeds them into a vector space, stores them in a local vector database, and allows the user to search those memories using natural language.

When the user asks a question, the system finds the most relevant past image and uses a Vision-Language Model to generate a helpful answer.

---

## Implementation Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Passive image capture | Done |
| 2 | Local image storage | Done |
| 3 | Metadata storage | Done |
| 4 | Image embedding (CLIP) | Done |
| 5 | Vector database (FAISS) | Done |
| 6 | Natural language query | Done |
| 7 | Image retrieval | Done |
| 8 | VLM reasoning | Done |

---

## Project Structure

```
VirtualMemoryAssistant/
├── main.py                 # Unified CLI entry point
├── config.py               # Central configuration (paths, constants, model IDs)
├── scripts/
│   ├── capture.py          # Webcam capture loop
│   ├── embed_images.py     # CLIP embedding pipeline (incremental)
│   ├── build_index.py      # FAISS index builder
│   └── query.py            # Natural language search + VLM answer
├── utils/
│   └── clip_utils.py       # Shared CLIP model loader
├── models/
│   └── vlm.py              # Moondream2 VLM wrapper
├── tests/                  # Full test suite (32 tests, all mocked)
├── data/
│   ├── raw/                # Captured JPEG frames (gitignored)
│   ├── metadata/           # Per-image metadata (gitignored)
│   │   └── captures.jsonl
│   └── embeddings/         # CLIP vectors + FAISS index (gitignored)
│       ├── image_embeddings.npy
│       ├── image_mapping.json
│       └── faiss.index
├── requirements.txt
└── start_venv.bat
```

---

## MVP Features

### 1. Passive Image Capture

The system captures images from a connected webcam at a fixed interval (default: every 10 seconds, up to `MAX_CAPTURES` frames).

```text
Capture one frame every 10 seconds
Save image locally to data/raw/
Append timestamp and file path to metadata
```

Each captured image becomes a memory entry.

### 2. Local Image Storage

Captured images are saved in:

```
data/raw/
```

Each image has a unique filename based on the timestamp:

```
frame_20260614_154210.jpg
```

### 3. Metadata Storage

Each captured image has a corresponding metadata record appended to a `.jsonl` file:

```
data/metadata/captures.jsonl
```

Each line is a self-contained JSON record:

```json
{
  "timestamp": "2026-06-14T15:42:10.123456",
  "image_path": "D:\\VirtualMemoryAssistant\\data\\raw\\frame_20260614_154210.jpg",
  "file_name": "frame_20260614_154210.jpg",
  "status": "captured"
}
```

### 4. Image Embedding

Each image is converted into a normalized embedding vector using CLIP (`openai/clip-vit-base-patch32`).

The pipeline (`scripts/embed_images.py`) does the following:

1. Reads `captures.jsonl` to find all recorded images
2. Loads each image with OpenCV and converts it to RGB PIL format
3. Passes each image through the CLIP image encoder
4. L2-normalizes the output feature vector
5. Saves all vectors and an index-to-metadata mapping:

```
data/embeddings/image_embeddings.npy   # shape: (N, 512)
data/embeddings/image_mapping.json     # list of {embedding_index, image_path, timestamp, file_name}
```

### 5. Vector Database

Image embeddings are loaded into a local FAISS flat inner-product index (`IndexFlatIP`).

Because the CLIP embeddings are L2-normalized, inner product search is equivalent to cosine similarity — no extra normalization step needed at query time.

The index is saved to:

```
data/embeddings/faiss.index
```

Library: `faiss-cpu`

### 6. Natural Language Query

The user submits a question in plain English. It is encoded with the CLIP text encoder (same model, same 512-dimensional space as the image embeddings) and L2-normalized before searching.

```
Where is my coffee mug?
```

### 7. Image Retrieval

The normalized text embedding is searched against the FAISS index using inner-product (cosine similarity). The top `TOP_K` matches are returned with their similarity scores:

```
Query: "Where is my coffee mug?"
----------------------------------------
#1  frame_20260614_154210.jpg
    Similarity : 0.8214
    Timestamp  : 2026-06-14T15:42:10.123456
    Path       : D:\VirtualMemoryAssistant\data\raw\frame_20260614_154210.jpg
```

### 8. Vision-Language Reasoning

The top retrieved image is passed to **Moondream2** (`vikhyatk/moondream2`) along with the user's question to generate a natural language answer. The model runs locally and is CUDA-accelerated on supported hardware (Jetson Orin Nano, discrete GPU).

Example:

```
Query: "Where is my coffee mug?"
----------------------------------------
#1  frame_20260614_154210.jpg
    Similarity : 0.8214
    ...

Answer: The coffee mug is on the desk to the left of the keyboard.
```

---

## MVP Logic Flow

```
1.  Webcam captures a frame every N seconds          (capture.py)
2.  Frame is saved to data/raw/                      (capture.py)
3.  Metadata is appended to captures.jsonl           (capture.py)
4.  Images are encoded into CLIP embeddings          (embed_images.py)
5.  Embeddings saved to data/embeddings/             (embed_images.py)
6.  Embeddings loaded into a FAISS index             (pending)
7.  User asks a natural language question            (pending)
8.  Question is embedded with CLIP text encoder      (pending)
9.  Vector search finds the most relevant image      (pending)
10. Retrieved image is passed to a VLM               (pending)
11. VLM answers the question using the image         (pending)
```

---

## Setup

### Prerequisites

- Python 3.9+
- A connected webcam

### Install dependencies

```bash
pip install -r requirements.txt
```

> **First-run note:** CLIP (`openai/clip-vit-base-patch32`) is ~600 MB and Moondream2 is ~2 GB. Both are downloaded automatically from HuggingFace on first use and cached locally. Ensure you have sufficient disk space and a network connection before running for the first time.

### Activate the virtual environment (Windows)

```bash
start_venv.bat
```

---

## Running

### Step 1 — Capture images

```bash
python scripts/capture.py
```

Captures `MAX_CAPTURES` frames from the default webcam, one every `CAPTURE_INTERVAL_SECONDS` seconds. Edit these constants at the top of the file to configure.

### Step 2 — Generate embeddings

```bash
python scripts/embed_images.py
```

Reads `data/metadata/captures.jsonl`, generates CLIP embeddings for all captured images, and saves them to `data/embeddings/`.

### Step 3 — Build FAISS index

```bash
python scripts/build_index.py
```

Loads `data/embeddings/image_embeddings.npy` and builds a FAISS flat inner-product index, saved to `data/embeddings/faiss.index`.

### Step 4 — Query

```bash
python scripts/query.py "where is my coffee mug?"
```

Or run without arguments to be prompted interactively. Returns the top 3 most similar image memories with similarity scores and timestamps.
