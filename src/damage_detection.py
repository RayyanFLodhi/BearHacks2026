import shutil
from pathlib import Path

import cv2
import numpy as np

from src.measurement import bbox_length_px, mask_area_px, skeleton_length_px
from src.utils import ensure_dir, safe_stem

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - keeps fallback available if ultralytics is missing.
    YOLO = None


# Tunable thresholds/constants.
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_CONF_THRESHOLD = 0.25

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
BLUR_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150
CRACK_MIN_AREA = 60
CRACK_MIN_LENGTH = 20
CRACK_MIN_ASPECT_RATIO = 2.0

DARK_THRESHOLD = 75
BLOB_MIN_AREA = 180
BLOB_MAX_CIRCULARITY = 0.75


def _run_yolo(image, detections, output_dir: Path):
    if YOLO is None:
        return

    try:
        local_model_path = output_dir / YOLO_MODEL_NAME
        model = YOLO(str(local_model_path) if local_model_path.exists() else YOLO_MODEL_NAME)
        downloaded_default_path = Path(YOLO_MODEL_NAME)
        if downloaded_default_path.exists() and not local_model_path.exists():
            shutil.move(str(downloaded_default_path), str(local_model_path))
        results = model(image, conf=YOLO_CONF_THRESHOLD, verbose=False)
        if not results:
            return
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w == 0 or h == 0:
                continue
            detections.append(
                {
                    "type": "generic_yolo",
                    "bbox": [x1, y1, w, h],
                    "length_px": bbox_length_px([x1, y1, w, h]),
                    "area_px": int(w * h),
                    "confidence": round(float(box.conf[0].item()), 4),
                    "source": "yolo",
                }
            )
    except Exception:
        # YOLO failure should not block OpenCV fallback.
        return


def detect_damage(image_path: str, use_yolo: bool = True) -> dict:
    image_path_obj = Path(image_path)
    image = cv2.imread(str(image_path_obj))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    output_dir = ensure_dir(Path("outputs"))
    image_stem = safe_stem(image_path_obj)
    crack_mask_path = output_dir / f"{image_stem}_crack_mask.png"
    blob_mask_path = output_dir / f"{image_stem}_blob_mask.png"

    h, w = image.shape[:2]
    detections = []

    if use_yolo:
        _run_yolo(image, detections, output_dir)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # A) Crack-like detection pipeline.
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, BLUR_KERNEL, 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    crack_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    crack_contours, _ = cv2.findContours(crack_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crack_mask = np.zeros_like(gray, dtype=np.uint8)
    for contour in crack_contours:
        area = cv2.contourArea(contour)
        if area < CRACK_MIN_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        length_px = max(bw, bh)
        if length_px < CRACK_MIN_LENGTH:
            continue
        aspect_ratio = (max(bw, bh) / max(1, min(bw, bh)))
        if aspect_ratio < CRACK_MIN_ASPECT_RATIO:
            continue

        cv2.drawContours(crack_mask, [contour], -1, 255, thickness=-1)
        contour_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        detections.append(
            {
                "type": "crack",
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "length_px": skeleton_length_px(contour_mask),
                "area_px": mask_area_px(contour_mask),
                "confidence": "heuristic",
                "source": "opencv_crack",
            }
        )

    # B) Pothole/chip-like darker, irregular blob detection.
    dark_regions = cv2.inRange(blurred, 0, DARK_THRESHOLD)
    dark_clean = cv2.morphologyEx(dark_regions, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    dark_clean = cv2.morphologyEx(dark_clean, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    blob_contours, _ = cv2.findContours(dark_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blob_mask = np.zeros_like(gray, dtype=np.uint8)
    for contour in blob_contours:
        area = cv2.contourArea(contour)
        if area < BLOB_MIN_AREA:
            continue
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter <= 0:
            continue
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity > BLOB_MAX_CIRCULARITY:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        cv2.drawContours(blob_mask, [contour], -1, 255, thickness=-1)
        detections.append(
            {
                "type": "pothole_or_chip",
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "length_px": None,
                "area_px": int(area),
                "confidence": "heuristic",
                "source": "opencv_blob",
            }
        )

    cv2.imwrite(str(crack_mask_path), crack_mask)
    cv2.imwrite(str(blob_mask_path), blob_mask)

    return {
        "image_path": str(image_path_obj),
        "image_width": int(w),
        "image_height": int(h),
        "detections": detections,
        "masks": {
            "crack_mask_path": str(crack_mask_path),
            "blob_mask_path": str(blob_mask_path),
        },
    }
