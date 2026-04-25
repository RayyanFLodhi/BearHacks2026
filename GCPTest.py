# GCPTest.py

from google.cloud import vision
from dotenv import load_dotenv
import os

load_dotenv()

IMAGE_PATH = "DroneImages/GeneralInfrastructure.png"


def detect_labels(image_path: str, max_results: int = 10):
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.label_detection(
        image=image,
        max_results=max_results
    )

    if response.error.message:
        raise Exception(response.error.message)

    return response.label_annotations


def classify_context(labels):
    surface_keywords = [
        "asphalt",
        "concrete",
        "road surface",
        "pavement",
        "sidewalk",
        "flooring",
        "brick",
        "wall"
    ]

    environment_keywords = [
        "residential area",
        "neighbourhood",
        "public space",
        "urban area",
        "suburb",
        "house",
        "home"
    ]

    scene_keywords = [
        "road",
        "sidewalk",
        "driveway",
        "street",
        "building",
        "wall"
    ]

    surface = None
    environment = None
    scene = None

    for label in labels:
        name = label.description.lower()

        if surface is None and any(keyword in name for keyword in surface_keywords):
            surface = {
                "label": label.description,
                "confidence": label.score
            }

        if environment is None and any(keyword in name for keyword in environment_keywords):
            environment = {
                "label": label.description,
                "confidence": label.score
            }

        if scene is None and any(keyword in name for keyword in scene_keywords):
            scene = {
                "label": label.description,
                "confidence": label.score
            }

    return surface, environment, scene


def main():
    labels = detect_labels(IMAGE_PATH)

    print(f"\nLabels for: {IMAGE_PATH}\n")

    for label in labels:
        print(f"{label.description}: {label.score:.2f}")

    surface, environment, scene = classify_context(labels)

    print("\n--- Structured Inspection Context ---")

    if surface:
        print(f"Surface Type: {surface['label']} ({surface['confidence']:.2f})")
    else:
        print("Surface Type: Unknown")

    if environment:
        print(f"Environment: {environment['label']} ({environment['confidence']:.2f})")
    else:
        print("Environment: Unknown")

    if scene:
        print(f"Scene/Object: {scene['label']} ({scene['confidence']:.2f})")
    else:
        print("Scene/Object: Unknown")


if __name__ == "__main__":
    main()