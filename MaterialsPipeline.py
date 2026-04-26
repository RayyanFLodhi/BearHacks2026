from pathlib import Path

import cv2
from dotenv import load_dotenv
from google.cloud import vision

load_dotenv()

VIDEO_PATH = Path("DroneImages") / "backyard.mp4"
SAMPLE_INTERVAL_SECONDS = 2
MAX_LABELS = 20

MATERIAL_KEYWORDS = (
    "wood",
    "metal",
    "plastic",
    "glass",
    "fabric",
    "concrete",
    "asphalt",
    "brick",
    "masonry",
    "stone",
    "ceramic",
    "tile",
    "gravel",
    "soil",
    "sand",
    "grass",
)

OBJECT_MATERIAL_HINTS = {
    "fence": ["wood", "metal", "vinyl"],
    "table": ["wood", "metal", "glass", "plastic"],
    "chair": ["wood", "metal", "plastic", "fabric"],
    "wall": ["brick", "concrete", "masonry", "drywall", "paint"],
    "building": ["brick", "concrete", "glass", "metal"],
    "road": ["asphalt", "concrete"],
    "sidewalk": ["concrete", "brick", "stone"],
    "ground": ["soil", "sand", "gravel"],
}


def frame_to_vision_image(frame):
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return vision.Image(content=buffer.tobytes())


def detect_scene(client, frame):
    image = frame_to_vision_image(frame)
    label_response = client.label_detection(image=image, max_results=MAX_LABELS)
    object_response = client.object_localization(image=image, max_results=10)

    if label_response.error.message:
        raise RuntimeError(f"Label detection error: {label_response.error.message}")
    if object_response.error.message:
        raise RuntimeError(f"Object localization error: {object_response.error.message}")

    labels = [
        {"name": label.description, "score": float(label.score)}
        for label in label_response.label_annotations
    ]
    objects = [
        {"name": obj.name, "score": float(obj.score)}
        for obj in object_response.localized_object_annotations
    ]
    return labels, objects


def extract_materials(labels, objects):
    matched_materials = []
    for item in labels:
        label_name = item["name"].lower()
        for keyword in MATERIAL_KEYWORDS:
            if keyword in label_name and keyword not in matched_materials:
                matched_materials.append(keyword)

    inferred_from_objects = {}
    for obj in objects:
        obj_name = obj["name"].lower()
        for key, hints in OBJECT_MATERIAL_HINTS.items():
            if key in obj_name:
                inferred_from_objects[obj_name] = hints

    return matched_materials, inferred_from_objects


def format_timestamp(seconds_total):
    mins = int(seconds_total // 60)
    secs = int(seconds_total % 60)
    return f"{mins:02d}:{secs:02d}"


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    sample_every_n_frames = max(1, int(round(fps * SAMPLE_INTERVAL_SECONDS)))
    client = vision.ImageAnnotatorClient()

    print(f"Analyzing video: {VIDEO_PATH}")
    print(f"Sampling every {SAMPLE_INTERVAL_SECONDS} seconds ({sample_every_n_frames} frames at ~{fps:.2f} FPS)")
    print("-" * 72)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_every_n_frames == 0:
            current_seconds = frame_idx / fps
            timestamp = format_timestamp(current_seconds)

            try:
                labels, objects = detect_scene(client, frame)
                materials, inferred = extract_materials(labels, objects)

                top_labels = ", ".join([f"{x['name']} ({x['score']:.2f})" for x in labels[:8]]) or "none"
                top_objects = ", ".join([f"{x['name']} ({x['score']:.2f})" for x in objects[:6]]) or "none"

                print(f"[{timestamp}]")
                print(f"  Objects: {top_objects}")
                print(f"  Labels: {top_labels}")
                print(f"  Materials (direct label match): {', '.join(materials) if materials else 'none'}")
                if inferred:
                    print(f"  Materials (object hints): {inferred}")
                print()
            except Exception as exc:
                print(f"[{timestamp}] GCP Vision error: {exc}")

        frame_idx += 1

    cap.release()
    print("Done.")


if __name__ == "__main__":
    main()
