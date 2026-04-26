from collections import defaultdict

import cv2
from dotenv import load_dotenv
from google.cloud import vision

load_dotenv()

# Camera and API behavior tuning.
CAMERA_INDEX = 1
FRAME_INTERVAL = 30  # Call GCP every ~1 second at 30 FPS.
MAX_LABELS = 12
MAX_OBJECTS = 10

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
)

OBJECT_MATERIAL_HINTS = {
    "person": ["fabric", "skin"],
    "human face": ["skin"],
    "chair": ["wood", "metal", "plastic", "fabric"],
    "table": ["wood", "metal", "glass", "plastic"],
    "bed": ["fabric", "wood", "metal"],
    "floor": ["wood", "tile", "concrete"],
    "wall": ["paint", "brick", "concrete", "drywall"],
    "road": ["asphalt", "concrete"],
    "sidewalk": ["concrete", "brick", "stone"],
}


def frame_to_vision_image(frame):
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return vision.Image(content=buffer.tobytes())


def detect_scene(frame, client):
    image = frame_to_vision_image(frame)

    label_response = client.label_detection(image=image, max_results=MAX_LABELS)
    object_response = client.object_localization(image=image, max_results=MAX_OBJECTS)

    if label_response.error.message:
        raise RuntimeError(f"Label detection error: {label_response.error.message}")
    if object_response.error.message:
        raise RuntimeError(f"Object localization error: {object_response.error.message}")

    labels = [
        {"name": item.description, "score": float(item.score)}
        for item in label_response.label_annotations
    ]
    objects = [
        {
            "name": item.name,
            "score": float(item.score),
            "vertices": [(v.x, v.y) for v in item.bounding_poly.normalized_vertices],
        }
        for item in object_response.localized_object_annotations
    ]
    return labels, objects


def extract_materials(labels, objects):
    label_names = [x["name"].lower() for x in labels]
    object_names = [x["name"].lower() for x in objects]

    found_materials = []
    for name in label_names:
        for keyword in MATERIAL_KEYWORDS:
            if keyword in name and keyword not in found_materials:
                found_materials.append(keyword)

    inferred = defaultdict(set)
    for obj_name in object_names:
        for key, hints in OBJECT_MATERIAL_HINTS.items():
            if key in obj_name:
                for hint in hints:
                    inferred[obj_name].add(hint)

    return found_materials, {k: sorted(v) for k, v in inferred.items()}


def draw_object_boxes(frame, objects):
    h, w = frame.shape[:2]
    for obj in objects:
        vertices = obj["vertices"]
        if len(vertices) < 2:
            continue
        xs = [int(x * w) for x, _ in vertices]
        ys = [int(y * h) for _, y in vertices]
        x1, x2 = max(0, min(xs)), min(w - 1, max(xs))
        y1, y2 = max(0, min(ys)), min(h - 1, max(ys))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 140, 0), 2)
        cv2.putText(
            frame,
            f"{obj['name']} ({obj['score']:.2f})",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 140, 0),
            1,
            cv2.LINE_AA,
        )


def draw_overlay(frame, labels, objects, materials):
    y = 25
    cv2.putText(frame, "GCP Vision Live Labels (press q to quit)", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y += 28
    top_objects = ", ".join([f"{x['name']}:{x['score']:.2f}" for x in objects[:4]]) or "none"
    cv2.putText(frame, f"Objects: {top_objects}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    y += 22
    top_labels = ", ".join([f"{x['name']}:{x['score']:.2f}" for x in labels[:4]]) or "none"
    cv2.putText(frame, f"Labels: {top_labels}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    y += 22
    mats = ", ".join(materials) if materials else "not confidently identified"
    cv2.putText(frame, f"Materials: {mats}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 0), 1)


def main():
    client = vision.ImageAnnotatorClient()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}.")

    cached_labels = []
    cached_objects = []
    cached_materials = []
    frame_count = 0

    print("Live feed started. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame read failed.")
            break

        if frame_count % FRAME_INTERVAL == 0:
            try:
                labels, objects = detect_scene(frame, client)
                materials, inferred_by_object = extract_materials(labels, objects)
                cached_labels = labels
                cached_objects = objects
                cached_materials = materials

                print("\n--- GCP Vision Frame Analysis ---")
                print("Objects:", ", ".join([f"{o['name']} ({o['score']:.2f})" for o in objects[:8]]) or "none")
                print("Labels:", ", ".join([f"{l['name']} ({l['score']:.2f})" for l in labels[:8]]) or "none")
                print("Materials (from labels):", ", ".join(materials) or "none")
                if inferred_by_object:
                    print("Likely materials by object:", dict(inferred_by_object))
            except Exception as exc:
                print(f"GCP Vision error: {exc}")

        draw_object_boxes(frame, cached_objects)
        draw_overlay(frame, cached_labels, cached_objects, cached_materials)
        cv2.imshow("GCP Vision Live Labeling", frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()