from pathlib import Path


def _severity(detections: list[dict]) -> str:
    crack_count = sum(1 for d in detections if d.get("type") == "crack")
    total_area = sum(int(d.get("area_px") or 0) for d in detections)
    if total_area >= 15000 or crack_count >= 6:
        return "HIGH"
    if total_area >= 3000 or len(detections) >= 2:
        return "MEDIUM"
    return "LOW"


def generate_basic_report(image_path: str, labels: list[dict], context: dict, damage_result: dict) -> str:
    image_name = Path(image_path).name
    top_labels = ", ".join([f"{x['label']} ({x['confidence']:.2f})" for x in labels[:5]]) or "N/A"

    detections = damage_result.get("detections", [])
    lines = [
        "AI Infrastructure Inspection Assistant - Basic CV Report",
        "=" * 56,
        f"Image analyzed: {image_name}",
        "",
        f"Top GCP labels: {top_labels}",
        f"Surface/material context: {context.get('surface_type')}",
        f"Scene/environment context: scene={context.get('scene')} | environment={context.get('environment')}",
        "",
        f"Detected damage candidates: {len(detections)}",
    ]

    if detections:
        lines.append("Per-detection summary:")
        for idx, det in enumerate(detections, start=1):
            lines.append(
                f"  {idx}. type={det.get('type')} | bbox={det.get('bbox')} | "
                f"length_px={det.get('length_px')} | area_px={det.get('area_px')}"
            )
    else:
        lines.append("Per-detection summary: none")

    lines.extend(
        [
            "",
            f"Estimated severity: {_severity(detections)}",
            "",
            "Limitations:",
            "- Measurements are pixel-based unless calibration marker or camera height is provided.",
            "",
            "Recommended next steps:",
            "- Capture additional angles/frames for cross-validation.",
            "- Add a known-size calibration marker for real-world dimensions.",
            "- Manually review flagged regions before maintenance decisions.",
            "",
            "# TODO: Integrate Gemma-generated natural language report once LLM stage is enabled.",
            "# TODO: Surface this report in dashboard/API layer when frontend/backend are added.",
        ]
    )
    return "\n".join(lines)
