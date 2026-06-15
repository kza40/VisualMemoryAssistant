# Continuously captures frames from a webcam at a fixed interval and writes each image
# to data/raw/ with a corresponding metadata entry in data/metadata/captures.jsonl.

from pathlib import Path
import cv2
from datetime import datetime
import json
import time



# Edit these to configure capture behaviour
CAMERA_ID = 0
CAPTURE_INTERVAL_SECONDS = 10
MAX_CAPTURES = 5  # set higher (or remove the limit) for real use

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_IMAGE_FOLDER = BASE_DIR / "data" / "raw"
METADATA_FOLDER = BASE_DIR / "data" / "metadata"
METADATA_FILE = METADATA_FOLDER / "captures.jsonl"
IMAGE_PREFIX = "frame"


def create_directories():
    """Create output directories if they don't exist."""
    RAW_IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)
    METADATA_FOLDER.mkdir(parents=True, exist_ok=True)


def open_camera( camera_id: int ):
    """Open the webcam and return the capture object, or raise if unavailable."""
    camera = cv2.VideoCapture(camera_id)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera with CAMERA_ID = {camera_id}")
    return camera


def generate_timestamps():
    """Return (filename_safe_timestamp, iso_timestamp) for the current moment."""
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S"), now.isoformat()


def save_frame( frame, filename_timestamp ):
    """Write a frame to data/raw/ and return its path."""
    filename = f"{IMAGE_PREFIX}_{filename_timestamp}.jpg"
    image_path = RAW_IMAGE_FOLDER / filename
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"Failed to save image to {image_path}")
    return image_path


def append_metadata( image_path, iso_timestamp ):
    """Append a single JSON record to the JSONL metadata file."""
    record = {
        "timestamp": iso_timestamp,
        "image_path": str(image_path),
        "file_name": image_path.name,
        "status": "captured",
    }
    with open(METADATA_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def capture_frame(camera, interval_seconds):
    """Capture one frame, persist it, then sleep for the configured interval."""
    success, frame = camera.read()

    if not success:
        print("Warning: Failed to capture frame. Skipping.")
        return

    filename_timestamp, iso_timestamp = generate_timestamps()

    try:
        image_path = save_frame(frame, filename_timestamp)
        append_metadata(image_path, iso_timestamp)
        print(f"\nSaved image: {image_path} at {iso_timestamp}")

    except Exception as error:
        print(f"Warning: {error}")

    time.sleep(interval_seconds)


def main():

    create_directories()
    camera = open_camera(CAMERA_ID)

    try:
        for _ in range(MAX_CAPTURES):
            capture_frame(camera, CAPTURE_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nCapture stopped by user.")

    finally:
        camera.release()
        print("\nCamera released. Exiting.")

if __name__ == "__main__":
    main()
