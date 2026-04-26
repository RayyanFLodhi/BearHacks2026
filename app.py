from __future__ import annotations

from datetime import datetime
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

import cv2
from flask import Flask, Response, jsonify, redirect, render_template, request, send_from_directory, url_for
from ollama import chat
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CAPTURES_DIR = STATIC_DIR / "captures"
RESULTS_DIR = STATIC_DIR / "results"
REPORTS_DIR = STATIC_DIR / "reports"

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_PDF_EXTENSIONS = {".pdf"}
FRAME_ANALYSIS_INTERVAL = 30
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
JPEG_QUALITY = 90

CAMERA_FALLBACK_INDICES = (1, 0, 2, 3)

_analysis_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gcp-analysis")
_analysis_lock = Lock()
_latest_boxes: list[dict] = []

def _camera_candidates(camera_index: int):
    backends = [cv2.CAP_DSHOW]
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    backends.append(cv2.CAP_ANY)
    for backend in backends:
        yield camera_index, backend

def analyze_with_gcp(frame) -> list[dict]:
    try:
        from camerafeed import detect_scene
        from google.cloud import vision
    except Exception:
        return []

    client = vision.ImageAnnotatorClient()
    _, objects = detect_scene(frame, client)

    h, w = frame.shape[:2]
    boxes: list[dict] = []
    for obj in objects:
        vertices = obj.get("vertices", [])
        if len(vertices) < 2:
            continue
        xs = [int(x * w) for x, _ in vertices]
        ys = [int(y * h) for _, y in vertices]
        boxes.append({
            "x1": max(0, min(xs)),
            "y1": max(0, min(ys)),
            "x2": min(w - 1, max(xs)),
            "y2": min(h - 1, max(ys)),
            "label": obj.get("name", "object"),
        })
    return boxes

def _open_camera_live(camera_index: int | None):
    if camera_index is not None:
        for idx, backend in _camera_candidates(camera_index):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap, idx
            cap.release()
        return None, None

    for idx in CAMERA_FALLBACK_INDICES:
        for candidate_idx, backend in _camera_candidates(idx):
            cap = cv2.VideoCapture(candidate_idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap, candidate_idx
            cap.release()
    return None, None

def _open_camera_demo():
    video_path = str(BASE_DIR / "DroneImages" / "backyard.mp4")
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        return cap, "backyard.mp4"
    return None, None

def _draw_boxes(frame, boxes: list[dict]) -> None:
    for box in boxes:
        x1, y1 = int(box.get("x1", 0)), int(box.get("y1", 0))
        x2, y2 = int(box.get("x2", 0)), int(box.get("y2", 0))
        label = str(box.get("label", "object"))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

def _stream_frames(is_demo: bool, camera_index: int | None = None):
    global _latest_boxes

    if is_demo:
        cap, used_index = _open_camera_demo()
    else:
        cap, used_index = _open_camera_live(camera_index)

    if cap is None or used_index is None:
        message = "Could not open camera source."
        yield b"--frame\r\nContent-Type: text/plain\r\n\r\n" + message.encode("utf-8") + b"\r\n"
        return

    frame_count = 0
    pending_analysis: Future | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_demo:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok: break
                else:
                    break
            
            if is_demo:
                frame = cv2.resize(frame, (CAPTURE_WIDTH, CAPTURE_HEIGHT))
                time.sleep(0.033)

            if pending_analysis and pending_analysis.done():
                try:
                    new_boxes = pending_analysis.result()
                    with _analysis_lock:
                        _latest_boxes = new_boxes
                except Exception:
                    pass
                finally:
                    pending_analysis = None

            if frame_count % FRAME_ANALYSIS_INTERVAL == 0 and pending_analysis is None:
                frame_for_analysis = frame.copy()
                pending_analysis = _analysis_executor.submit(analyze_with_gcp, frame_for_analysis)

            with _analysis_lock:
                boxes_snapshot = list(_latest_boxes)
            _draw_boxes(frame, boxes_snapshot)

            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                frame_count += 1
                continue

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            frame_count += 1
    finally:
        cap.release()

def ensure_directories() -> None:
    for folder in (CAPTURES_DIR, RESULTS_DIR, REPORTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

def list_files(folder: Path, allowed_extensions: set[str]) -> list[str]:
    files = [path.name for path in folder.iterdir() if path.is_file() and path.suffix.lower() in allowed_extensions]
    return sorted(files, reverse=True)

def detection_label(filename: str) -> str:
    stem = Path(filename).stem
    if "_" in stem:
        prefix = stem.split("_", 1)[0]
        if prefix:
            return prefix.replace("-", " ").title()
    return "Inspection Result"

@app.route("/")
def index():
    ensure_directories()
    captures = list_files(CAPTURES_DIR, ALLOWED_IMAGE_EXTENSIONS)
    results = list_files(RESULTS_DIR, ALLOWED_IMAGE_EXTENSIONS)
    reports = list_files(REPORTS_DIR, ALLOWED_PDF_EXTENSIONS)
    result_items = [{"filename": name, "label": detection_label(name)} for name in results]
    return render_template("index.html", captures=captures, results=result_items, reports=reports)

@app.route("/demo")
def demo():
    ensure_directories()
    captures = list_files(CAPTURES_DIR, ALLOWED_IMAGE_EXTENSIONS)
    results = list_files(RESULTS_DIR, ALLOWED_IMAGE_EXTENSIONS)
    reports = list_files(REPORTS_DIR, ALLOWED_PDF_EXTENSIONS)
    result_items = [{"filename": name, "label": detection_label(name)} for name in results]
    return render_template("demo_index.html", captures=captures, results=result_items, reports=reports)

@app.route("/video_feed")
def video_feed():
    camera_index_param = request.args.get("camera_index", default=None, type=int)
    return Response(_stream_frames(is_demo=False, camera_index=camera_index_param), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_demo")
def video_feed_demo():
    return Response(_stream_frames(is_demo=True), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/capture", methods=["POST"])
def capture():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({"status": "ok", "message": f"Capture requested at {timestamp}."})

@app.route("/chat_gemma", methods=["POST"])
def chat_gemma():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"status": "error", "error": "Message cannot be empty."}), 400

    try:
        SYSTEM_PROMPT = """You are a professional structural inspector.

Analyze the following observation and generate a concise inspection report.

Observation:
- Material: Brick wall
- Issue detected: 200+ Cracks, holes, and surface chipping detected by a computer vision model
- Location: Exterior wall
- Context: Captured from drone inspection

Write a structured report with these sections:

1. Summary
2. Observations
3. Possible Causes
4. Severity (Low/Medium/High with reasoning)
5. Recommended Actions
6. Next Steps

Keep the tone professional and clear. Avoid unnecessary fluff. Assume the reader is a homeowner or property manager.
Do NOT add a 'Date of Observation'.
"""

        response = chat(
            model="gemma4:e4b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ],
        )
        assistant_reply = (response.message.content or "").strip()
        if not assistant_reply:
            assistant_reply = "Gemma returned an empty response."
        return jsonify({"status": "ok", "reply": assistant_reply})
    except Exception as exc:
        return jsonify({"status": "error", "error": f"Gemma request failed: {exc}"}), 500

@app.route("/upload_result", methods=["POST"])
def upload_result():
    ensure_directories()
    file = request.files.get("result_file")
    if not file or file.filename == "":
        return redirect(url_for("index"))
    filename = secure_filename(file.filename)
    if Path(filename).suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        return redirect(url_for("index"))
    file.save(RESULTS_DIR / filename)
    return redirect(url_for("index"))

@app.route("/upload_report", methods=["POST"])
def upload_report():
    ensure_directories()
    file = request.files.get("report_file")
    if not file or file.filename == "":
        return redirect(url_for("index"))
    filename = secure_filename(file.filename)
    if Path(filename).suffix.lower() not in ALLOWED_PDF_EXTENSIONS:
        return redirect(url_for("index"))
    file.save(REPORTS_DIR / filename)
    return redirect(url_for("index"))

@app.route("/download_report/<path:filename>")
def download_report(filename: str):
    return send_from_directory(REPORTS_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    ensure_directories()
    app.run(debug=True, port=5000)
