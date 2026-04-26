from pathlib import Path

from google.cloud import vision


SURFACE_KEYWORDS = (
    "asphalt",
    "concrete",
    "brick",
    "road surface",
    "pavement",
    "sidewalk",
    "wall",
    "masonry",
)
ENVIRONMENT_KEYWORDS = (
    "residential area",
    "neighbourhood",
    "public space",
    "urban area",
    "suburb",
    "house",
    "home",
)
SCENE_KEYWORDS = (
    "road",
    "sidewalk",
    "driveway",
    "street",
    "building",
    "wall",
)


def detect_labels(image_path: str, max_results: int = 10) -> list[dict]:
    image_bytes = Path(image_path).read_bytes()
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.label_detection(image=image, max_results=max_results)
    if response.error.message:
        raise RuntimeError(f"GCP Vision label_detection failed: {response.error.message}")

    labels = []
    for label in response.label_annotations:
        labels.append(
            {
                "label": label.description,
                "confidence": round(float(label.score), 4),
            }
        )
    return labels


def _pick_best(labels: list[dict], keywords: tuple[str, ...]):
    for item in sorted(labels, key=lambda x: x.get("confidence", 0), reverse=True):
        value = item.get("label", "").lower()
        if any(keyword in value for keyword in keywords):
            return item
    return None


def classify_context(labels: list[dict]) -> dict:
    return {
        "surface_type": _pick_best(labels, SURFACE_KEYWORDS),
        "environment": _pick_best(labels, ENVIRONMENT_KEYWORDS),
        "scene": _pick_best(labels, SCENE_KEYWORDS),
    }
