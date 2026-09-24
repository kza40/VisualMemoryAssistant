# Visual Memory Assistant

[![tests](https://github.com/kza40/VirtualMemoryAssistant/actions/workflows/tests.yml/badge.svg)](https://github.com/kza40/VirtualMemoryAssistant/actions/workflows/tests.yml)

A camera on a Jetson Orin Nano takes a photo every few seconds. Later you can ask it
something like "where did I leave my keys?" and it finds the most relevant frame and
answers with a small vision-language model. Everything runs on the device.

```
camera ──> CLIP image encoder ──> FAISS index
                                      │
question ──> CLIP text encoder ───────┴──> best frame ──> Moondream2 ──> answer
```

## Status

The whole pipeline is written and covered by unit tests (34, with the camera and models
mocked, run in CI). It has not been run end to end on the Orin Nano yet. That's the next
step, and latency and memory numbers will go here once I have them.

## How it works

- `scripts/capture.py` grabs a frame every N seconds from a USB webcam or the Jetson CSI
  camera (through GStreamer) and appends a record to `data/metadata/captures.jsonl`.
- `scripts/embed_images.py` encodes new frames with CLIP ViT-B/32. It's incremental, so
  frames that are already embedded get skipped.
- `scripts/build_index.py` loads the vectors into a flat inner-product FAISS index. The
  vectors are L2-normalized, so inner product is cosine similarity.
- `scripts/query.py` embeds the question with CLIP's text encoder, pulls the top 3
  frames, and gives the best one to Moondream2 along with the question.

## Why these choices

- **CLIP ViT-B/32**: puts images and text in the same space, so there's no captioning
  step, and it's small enough to share the Nano's 8 GB with a VLM.
- **Flat FAISS index**: exact search is fast at this scale. A full day at one frame
  every 10 s is about 8,600 vectors. Much past that I'd switch to an IVF index.
- **Moondream2**: one of the few VLMs small enough (~2B params) for an edge board. It's
  pinned to a fixed revision because it loads remote code.

## Limitations

- Only the top frame goes to the VLM, so if retrieval picks the wrong frame the answer is
  wrong too.
- Ranking ignores time. "Where did I *last* see X" really means "which frame looks most
  like X".
- CLIP matches descriptions ("a photo of a mug") better than questions, and the question
  is passed in as-is.
- No retention policy yet. Frames pile up until you delete them.

## Setup (Jetson Orin Nano, JetPack 6)

Install NVIDIA's torch wheel first (instructions are in `requirements_jetson.txt`), then:

```bash
sudo apt install python3-opencv
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements_jetson.txt
```

The venv uses `--system-site-packages` so it picks up JetPack's OpenCV, which is built
with GStreamer. The pip OpenCV wheel isn't, and the CSI camera won't open without it.
If you're using a CSI camera, set `CAMERA_SOURCE = "csi"` in `config.py`.

CLIP and Moondream2 (about 2.6 GB together) download from Hugging Face on first use.
To start capturing on boot, see `jetson/capture.service`.

## Usage

```bash
python main.py capture                        # take frames (interval and count in config.py)
python main.py embed                          # embed any new frames
python main.py index                          # rebuild the FAISS index
python main.py ask "where is my coffee mug?"
```

## Tests

The camera and both models are mocked, so the tests run on any machine without a GPU:

```bash
pip install -r requirements.txt
pytest
ruff check . && ruff format --check .   # lint, same as CI
```

## Privacy

Frames, embeddings and answers stay on the device. Nothing is uploaded anywhere.

## License

MIT
