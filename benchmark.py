import cv2
import time
import os
import csv
from ultralytics import YOLO

# ==========================================
# VISIONTRACK AI
# AUTOMATIC MODEL BENCHMARK
# ==========================================

VIDEO_PATH = "input/test_video.mp4"
MODEL_PATH = "yolo11n.pt"

IMAGE_SIZES = [640, 512, 416, 320]

OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("VISIONTRACK AI - MODEL PERFORMANCE BENCHMARK")
print("=" * 60)

print()
print(f"Video: {VIDEO_PATH}")
print(f"Model: {MODEL_PATH}")
print(f"Image sizes: {IMAGE_SIZES}")
print()

# ==========================================
# CHECK VIDEO
# ==========================================

if not os.path.exists(VIDEO_PATH):

    print("ERROR: Video not found.")
    print(f"Expected: {VIDEO_PATH}")
    exit()

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading YOLO11n...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print()

# ==========================================
# RESULTS
# ==========================================

benchmark_results = []

# ==========================================
# TEST EACH IMAGE SIZE
# ==========================================

for image_size in IMAGE_SIZES:

    print("-" * 60)
    print(f"Testing inference size: {image_size}")
    print("-" * 60)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():

        print("ERROR: Could not open video.")
        continue

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    video_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    # --------------------------------------
    # Limit benchmark frames
    # --------------------------------------

    benchmark_frames = min(
        total_frames,
        100
    )

    frame_count = 0
    total_inference_time = 0
    total_runtime_start = time.perf_counter()

    while frame_count < benchmark_frames:

        ret, frame = cap.read()

        if not ret:
            break

        inference_start = time.perf_counter()

        model.predict(
            frame,
            imgsz=image_size,
            conf=0.5,
            device="cpu",
            verbose=False
        )

        inference_end = time.perf_counter()

        total_inference_time += (
            inference_end -
            inference_start
        )

        frame_count += 1

    total_runtime = (
        time.perf_counter()
        - total_runtime_start
    )

    cap.release()

    # --------------------------------------
    # Calculate metrics
    # --------------------------------------

    if frame_count > 0:

        average_inference_ms = (
            total_inference_time /
            frame_count
        ) * 1000

        average_fps = (
            frame_count /
            total_runtime
        )

    else:

        average_inference_ms = 0
        average_fps = 0

    result = {
        "image_size": image_size,
        "frames_tested": frame_count,
        "average_fps": round(
            average_fps,
            2
        ),
        "average_inference_ms": round(
            average_inference_ms,
            2
        ),
        "runtime_seconds": round(
            total_runtime,
            2
        )
    }

    benchmark_results.append(result)

    print(
        f"Frames tested: {frame_count}"
    )

    print(
        f"Average FPS: {average_fps:.2f}"
    )

    print(
        f"Average inference: "
        f"{average_inference_ms:.2f} ms"
    )

    print(
        f"Runtime: "
        f"{total_runtime:.2f} sec"
    )

    print()

# ==========================================
# FIND BEST SETTINGS
# ==========================================

if not benchmark_results:

    print("No benchmark results.")
    exit()

best_fps_result = max(
    benchmark_results,
    key=lambda x: x["average_fps"]
)

best_latency_result = min(
    benchmark_results,
    key=lambda x: x["average_inference_ms"]
)

# ==========================================
# SAVE CSV
# ==========================================

csv_path = (
    f"{OUTPUT_DIR}/model_benchmark.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image_size",
            "frames_tested",
            "average_fps",
            "average_inference_ms",
            "runtime_seconds"
        ]
    )

    writer.writeheader()

    writer.writerows(
        benchmark_results
    )

# ==========================================
# FINAL REPORT
# ==========================================

print("=" * 60)
print("VISIONTRACK AI - BENCHMARK COMPLETE")
print("=" * 60)

print()

print(
    f"{'Image Size':<15}"
    f"{'FPS':<15}"
    f"{'Inference':<20}"
)

print("-" * 50)

for result in benchmark_results:

    print(
        f"{result['image_size']:<15}"
        f"{result['average_fps']:<15}"
        f"{result['average_inference_ms']:.2f} ms"
    )

print()

print(
    f"Best FPS: "
    f"{best_fps_result['image_size']} "
    f"({best_fps_result['average_fps']:.2f} FPS)"
)

print(
    f"Lowest latency: "
    f"{best_latency_result['image_size']} "
    f"({best_latency_result['average_inference_ms']:.2f} ms)"
)

print()

print(
    f"Benchmark saved to: {csv_path}"
)

print("=" * 60)