import json
from pathlib import Path

from dotenv import load_dotenv

from src.damage_detection import detect_damage
from src.gcp_vision import classify_context, detect_labels
from src.report_generator import generate_basic_report
from src.utils import ensure_dir, list_image_files, safe_stem
from src.visualization import draw_detections

PROCESS_ALL = True
DEFAULT_IMAGE = Path("DroneImages") / "GeneralInfrastructure.png"
OUTPUTS_DIR = Path("outputs")


def process_image(image_path: Path):
    labels = []
    context = {"surface_type": None, "environment": None, "scene": None}

    try:
        labels = detect_labels(str(image_path))
        context = classify_context(labels)
    except Exception as exc:
        print(f"[WARN] GCP Vision failed for {image_path.name}: {exc}")
        print("[WARN] Continuing with local CV-only damage detection.")

    damage_result = detect_damage(str(image_path), use_yolo=True)

    stem = safe_stem(image_path)
    annotated_path = OUTPUTS_DIR / f"{stem}_annotated.png"
    report_path = OUTPUTS_DIR / f"{stem}_report.txt"
    results_json_path = OUTPUTS_DIR / f"{stem}_results.json"

    draw_detections(str(image_path), damage_result.get("detections", []), str(annotated_path))
    report = generate_basic_report(str(image_path), labels, context, damage_result)
    report_path.write_text(report, encoding="utf-8")

    full_result = {
        "image_path": str(image_path),
        "gcp_labels": labels,
        "context": context,
        "damage_result": damage_result,
        "outputs": {
            "annotated_image": str(annotated_path),
            "report_txt": str(report_path),
            "results_json": str(results_json_path),
        },
        # TODO: Add Gemma summary field once LLM stage is enabled.
    }
    results_json_path.write_text(json.dumps(full_result, indent=2), encoding="utf-8")

    top_labels = ", ".join([x["label"] for x in labels[:3]]) if labels else "N/A"
    surface_label = context.get("surface_type", {}).get("label") if context.get("surface_type") else "Unknown"
    print(f"[OK] {image_path.name}")
    print(f"  top labels: {top_labels}")
    print(f"  surface type: {surface_label}")
    print(f"  detections: {len(damage_result.get('detections', []))}")
    print(f"  annotated: {annotated_path}")
    print(f"  report: {report_path}")
    print(f"  results: {results_json_path}")


def main():
    load_dotenv()
    ensure_dir(OUTPUTS_DIR)

    if PROCESS_ALL:
        image_paths = list_image_files("DroneImages")
    else:
        image_paths = [DEFAULT_IMAGE] if DEFAULT_IMAGE.exists() else []

    if not image_paths:
        print("[INFO] No images found in DroneImages/.")
        return

    print(f"[INFO] Processing {len(image_paths)} image(s)...")
    for image_path in image_paths:
        try:
            process_image(Path(image_path))
        except Exception as exc:
            print(f"[ERROR] Failed to process {image_path}: {exc}")


if __name__ == "__main__":
    main()
