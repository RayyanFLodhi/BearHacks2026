import cv2
import tempfile
from google.cloud import vision
from dotenv import load_dotenv
import os

# Load credentials
load_dotenv()

# Initialize Vision API client ONCE (important)
client = vision.ImageAnnotatorClient()

FRAME_INTERVAL = 60  # send 1 frame every 60 frames (~2 sec)

def detect_labels_from_frame(frame):
    # Save frame temporarily
    _, buffer = cv2.imencode(".jpg", frame)
    content = buffer.tobytes()

    image = vision.Image(content=content)

    response = client.label_detection(
        image=image,
        max_results=10
    )

    if response.error.message:
        print("API Error:", response.error.message)
        return []

    return response.label_annotations


def classify_context(labels):
    surface_keywords = ["asphalt", "concrete", "road surface", "pavement", "sidewalk"]
    environment_keywords = ["residential area", "neighbourhood", "public space"]
    scene_keywords = ["road", "sidewalk", "driveway"]

    surface = None
    environment = None
    scene = None

    for label in labels:
        name = label.description.lower()

        if not surface and any(k in name for k in surface_keywords):
            surface = label.description

        if not environment and any(k in name for k in environment_keywords):
            environment = label.description

        if not scene and any(k in name for k in scene_keywords):
            scene = label.description

    return surface, environment, scene


# --- MAIN LOOP ---

cap = cv2.VideoCapture(1)  # OBS virtual camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Drone Feed", frame)

    # Every N frames → call Vision API
    if frame_count % FRAME_INTERVAL == 0:
        print("\n--- Sending frame to Vision API ---")

        labels = detect_labels_from_frame(frame)

        for label in labels:
            print(f"{label.description}: {label.score:.2f}")

        surface, environment, scene = classify_context(labels)

        print("\nStructured Context:")
        print("Surface:", surface)
        print("Environment:", environment)
        print("Scene:", scene)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()