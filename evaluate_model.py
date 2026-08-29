import cv2
import os
import csv
import time
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================

VIDEO_PATH = "input/test_video.mp4"
MODEL_PATH = "yolo11n.pt"

IMAGE_SIZE = 320
CONFIDENCE = 0.50

MAX_FRAMES = 100

OUTPUT_FILE = "results/evaluation_metrics.csv"

# ==========================================
# HEADER
# ==========================================

print("=" * 60)
print("VISIONTRACK AI")
print("MODEL EVALUATION")
print("=" * 60)

# ==========================================
# CHECK FILES
# ==========================================

if not os.path.exists(VIDEO_PATH):

    print()
    print("ERROR: Video not found.")
    print(f"Expected: {VIDEO_PATH}")
    exit()

if not os.path.exists(MODEL_PATH):

    print()
    print("ERROR: YOLO model not found.")
    print(f"Expected: {MODEL_PATH}")
    exit()

# ==========================================
# LOAD MODEL
# ==========================================

print()
print("Loading YOLO11n...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")

# ==========================================
# OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("ERROR: Could not open video.")
    exit()

total_video_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

video_fps = cap.get(
    cv2.CAP_PROP_FPS
)

print()
print(f"Video frames: {total_video_frames}")
print(f"Video FPS: {video_fps:.2f}")
print(f"Evaluation frames: {MAX_FRAMES}")
print(f"Inference size: {IMAGE_SIZE}")
print(f"Confidence: {CONFIDENCE}")

# ==========================================
# METRICS
# ==========================================

frames_processed = 0

total_inference_time = 0

total_objects = 0

class_counts = {
    "person": 0,
    "car": 0,
    "truck": 0,
    "bus": 0,
    "motorcycle": 0,
    "bicycle": 0
}

confidence_values = []

# ==========================================
# EVALUATION
# ==========================================

print()
print("-" * 60)
print("Running evaluation...")
print("-" * 60)

while frames_processed < MAX_FRAMES:

    ret, frame = cap.read()

    if not ret:
        break

    frames_processed += 1

    start = time.perf_counter()

    results = model.predict(
        frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device=(
            0
            if __import__("torch").cuda.is_available()
            else "cpu"
        ),
        verbose=False
    )

    inference_time = (
        time.perf_counter() - start
    )

    total_inference_time += inference_time

    result = results[0]

    if result.boxes is not None:

        boxes = result.boxes

        if boxes.cls is not None:

            classes = boxes.cls.int().tolist()

            total_objects += len(classes)

            for class_id in classes:

                class_name = model.names[class_id]

                if class_name in class_counts:

                    class_counts[
                        class_name
                    ] += 1

        if boxes.conf is not None:

            confidence_values.extend(
                boxes.conf.tolist()
            )

# ==========================================
# CLOSE VIDEO
# ==========================================

cap.release()

# ==========================================
# CALCULATE METRICS
# ==========================================

if frames_processed > 0:

    average_inference_ms = (
        total_inference_time /
        frames_processed
    ) * 1000

    inference_fps = (
        frames_processed /
        total_inference_time
        if total_inference_time > 0
        else 0
    )

    objects_per_frame = (
        total_objects /
        frames_processed
    )

else:

    average_inference_ms = 0
    inference_fps = 0
    objects_per_frame = 0

if confidence_values:

    average_confidence = (
        sum(confidence_values) /
        len(confidence_values)
    )

    minimum_confidence = min(
        confidence_values
    )

    maximum_confidence = max(
        confidence_values
    )

else:

    average_confidence = 0
    minimum_confidence = 0
    maximum_confidence = 0

# ==========================================
# SAVE RESULTS
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)

rows = [

    ["Metric", "Value"],

    ["Frames Evaluated", frames_processed],

    [
        "Inference Size",
        IMAGE_SIZE
    ],

    [
        "Confidence Threshold",
        CONFIDENCE
    ],

    [
        "Average Inference (ms)",
        round(
            average_inference_ms,
            2
        )
    ],

    [
        "Inference FPS",
        round(
            inference_fps,
            2
        )
    ],

    [
        "Total Objects Detected",
        total_objects
    ],

    [
        "Objects Per Frame",
        round(
            objects_per_frame,
            2
        )
    ],

    [
        "Average Detection Confidence",
        round(
            average_confidence,
            4
        )
    ],

    [
        "Minimum Detection Confidence",
        round(
            minimum_confidence,
            4
        )
    ],

    [
        "Maximum Detection Confidence",
        round(
            maximum_confidence,
            4
        )
    ],

    ["Persons Detected", class_counts["person"]],

    ["Cars Detected", class_counts["car"]],

    ["Trucks Detected", class_counts["truck"]],

    ["Buses Detected", class_counts["bus"]],

    [
        "Motorcycles Detected",
        class_counts["motorcycle"]
    ],

    [
        "Bicycles Detected",
        class_counts["bicycle"]
    ]
]

with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerows(rows)

# ==========================================
# DISPLAY
# ==========================================

print()
print("=" * 60)
print("VISIONTRACK AI - EVALUATION COMPLETE")
print("=" * 60)

print()

print(
    f"Frames evaluated: {frames_processed}"
)

print(
    f"Average inference: "
    f"{average_inference_ms:.2f} ms"
)

print(
    f"Inference FPS: "
    f"{inference_fps:.2f}"
)

print(
    f"Total objects detected: "
    f"{total_objects}"
)

print(
    f"Objects per frame: "
    f"{objects_per_frame:.2f}"
)

print(
    f"Average detection confidence: "
    f"{average_confidence:.3f}"
)

print()

print("Class detections:")
print(
    f"Persons: {class_counts['person']}"
)
print(
    f"Cars: {class_counts['car']}"
)
print(
    f"Trucks: {class_counts['truck']}"
)
print(
    f"Buses: {class_counts['bus']}"
)
print(
    f"Motorcycles: {class_counts['motorcycle']}"
)
print(
    f"Bicycles: {class_counts['bicycle']}"
)

print()

print(
    f"Evaluation saved to: {OUTPUT_FILE}"
)

print("=" * 60)