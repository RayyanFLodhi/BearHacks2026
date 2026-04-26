from pathlib import Path

import cv2

from src.utils import ensure_dir


COLORS = {
    "crack": (0, 0, 255),  # red in BGR
    "pothole_or_chip": (0, 165, 255),  # orange
    "generic_yolo": (255, 0, 0),  # blue
}


def draw_detections(image_path: str, detections: list[dict], output_path: str) -> str:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    for det in detections:
        x, y, w, h = det.get("bbox", [0, 0, 0, 0])
        det_type = det.get("type", "unknown")
        color = COLORS.get(det_type, (255, 255, 255))
        label = f"{det_type} | len={det.get('length_px')}px | area={det.get('area_px')}px"
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            image,
            label,
            (x, max(15, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    out_path = Path(output_path)
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), image)
    return str(out_path)
