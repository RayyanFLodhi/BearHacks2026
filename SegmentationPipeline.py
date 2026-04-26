import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = Path("DroneImages") / "backyard.mp4"
OUTPUT_DIR = Path("outputs") / "segmentation_backyard"
FRAME_INTERVAL_SECONDS = 2

# Segmentation / detection tuning.
SEG_MODEL_NAME = "yolov8n-seg.pt"
SEG_CONF = 0.25
# Higher thresholds to keep only larger, clearer wall hole candidates.
MIN_HOLE_AREA = 260
MIN_BLOB_BBOX_SIDE = 18
MAX_BLOB_CIRCULARITY = 0.65


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_ts(seconds_total: float) -> str:
    mins = int(seconds_total // 60)
    secs = int(seconds_total % 60)
    return f"{mins:02d}_{secs:02d}"


def init_segmentation_model():
    try:
        return YOLO(SEG_MODEL_NAME)
    except Exception:
        return None


def get_ignore_mask_from_segmentation(model, frame):
    """
    Build an ignore-mask for large foreground objects (hose reel etc.)
    so wall damage search focuses on likely wall pixels.
    """
    h, w = frame.shape[:2]
    ignore_mask = np.zeros((h, w), dtype=np.uint8)
    if model is None:
        return ignore_mask

    try:
        results = model(frame, conf=SEG_CONF, verbose=False)
        if not results:
            return ignore_mask
        result = results[0]
        if result.masks is None:
            return ignore_mask

        masks = result.masks.data.cpu().numpy()
        for m in masks:
            binary = (m > 0.5).astype(np.uint8) * 255
            resized = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
            ignore_mask = cv2.bitwise_or(ignore_mask, resized)
    except Exception:
        # If segmentation fails, fallback still runs on entire frame.
        return np.zeros((h, w), dtype=np.uint8)

    return ignore_mask


def detect_hole_regions(frame, analysis_mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    hole_mask = np.zeros_like(gray, dtype=np.uint8)
    detections = []

    # Hole-only: dark irregular blobs in wall area.
    dark = cv2.inRange(blurred, 0, 80)
    dark = cv2.bitwise_and(dark, analysis_mask)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8), iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8), iterations=1)
    blob_contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in blob_contours:
        area = cv2.contourArea(cnt)
        if area < MIN_HOLE_AREA:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity > MAX_BLOB_CIRCULARITY:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if max(w, h) < MIN_BLOB_BBOX_SIDE:
            continue
        cv2.drawContours(hole_mask, [cnt], -1, 255, -1)
        detections.append(
            {
                "type": "hole",
                "bbox": [int(x), int(y), int(w), int(h)],
                "length_px": int(max(w, h)),
                "area_px": int(area),
            }
        )

    return detections, hole_mask


def draw_detections(frame, detections):
    out = frame.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        color = (0, 165, 255)
        label = f"{det['type']} | len={det['length_px']}px | area={det['area_px']}px"
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    frame_dir = ensure_dir(OUTPUT_DIR / "frames")
    mask_dir = ensure_dir(OUTPUT_DIR / "masks")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    sample_every_n = max(1, int(round(fps * FRAME_INTERVAL_SECONDS)))

    seg_model = init_segmentation_model()
    mode = "YOLO segmentation + OpenCV damage analysis" if seg_model is not None else "OpenCV-only damage analysis"
    print(f"Processing {VIDEO_PATH} every {FRAME_INTERVAL_SECONDS}s ({sample_every_n} frames) using: {mode}")

    frame_idx = 0
    analyzed_frames = []
    total_damage_count = 0
    total_damage_area_px = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_every_n == 0:
            seconds = frame_idx / fps
            ts = format_ts(seconds)

            ignore_mask = get_ignore_mask_from_segmentation(seg_model, frame)
            analysis_mask = cv2.bitwise_not(ignore_mask)
            detections, hole_mask = detect_hole_regions(frame, analysis_mask)
            annotated = draw_detections(frame, detections)

            annotated_name = f"frame_{ts}_annotated.png"
            hole_name = f"frame_{ts}_hole_mask.png"
            cv2.imwrite(str(frame_dir / annotated_name), annotated)
            cv2.imwrite(str(mask_dir / hole_name), hole_mask)

            frame_area = int(sum(d["area_px"] for d in detections))
            total_damage_count += len(detections)
            total_damage_area_px += frame_area

            frame_result = {
                "timestamp_seconds": round(seconds, 2),
                "frame_index": int(frame_idx),
                "damage_count": len(detections),
                "damage_area_px": frame_area,
                "detections": detections,
                "outputs": {
                    "annotated_frame": str(frame_dir / annotated_name),
                    "hole_mask": str(mask_dir / hole_name),
                },
            }
            analyzed_frames.append(frame_result)

            print(
                f"[{ts.replace('_', ':')}] "
                f"damage_count={len(detections)} "
                f"damage_area_px={frame_area}"
            )

        frame_idx += 1

    cap.release()

    summary = {
        "video_path": str(VIDEO_PATH),
        "frame_interval_seconds": FRAME_INTERVAL_SECONDS,
        "total_sampled_frames": len(analyzed_frames),
        "total_damage_count": int(total_damage_count),
        "total_damage_area_px": int(total_damage_area_px),
        "frames": analyzed_frames,
    }

    ensure_dir(OUTPUT_DIR)
    json_path = OUTPUT_DIR / "results.json"
    txt_path = OUTPUT_DIR / "summary.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    txt_path.write_text(
        "\n".join(
            [
                f"Video: {VIDEO_PATH}",
                f"Frame interval (s): {FRAME_INTERVAL_SECONDS}",
                f"Total sampled frames: {len(analyzed_frames)}",
                f"Total damage count: {total_damage_count}",
                f"Total damage area (px): {total_damage_area_px}",
                f"Annotated frames folder: {frame_dir}",
                f"Masks folder: {mask_dir}",
                "",
                "Use frame images in the website gallery. Parse results.json for metrics/cards.",
            ]
        ),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Total damage count: {total_damage_count}")
    print(f"Total damage area (px): {total_damage_area_px}")
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
