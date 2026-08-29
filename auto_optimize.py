import pandas as pd
import os

BENCHMARK_FILE = "results/model_benchmark.csv"

print("=" * 60)
print("VISIONTRACK AI - AUTOMATIC OPTIMIZATION")
print("=" * 60)

if not os.path.exists(BENCHMARK_FILE):
    print()
    print("ERROR: Benchmark file not found.")
    print()
    print("Run this first:")
    print("python benchmark.py")
    exit()

df = pd.read_csv(BENCHMARK_FILE)

# ==========================================
# FIND BEST CONFIGURATION
# ==========================================

best_fps_row = df.loc[
    df["average_fps"].idxmax()
]

best_latency_row = df.loc[
    df["average_inference_ms"].idxmin()
]

best_size = int(
    best_fps_row["image_size"]
)

best_fps = float(
    best_fps_row["average_fps"]
)

best_latency = float(
    best_latency_row["average_inference_ms"]
)

# ==========================================
# SAVE CONFIGURATION
# ==========================================

config = {
    "model": "YOLO11n",
    "optimized_image_size": best_size,
    "best_fps": round(best_fps, 2),
    "lowest_latency_ms": round(best_latency, 2),
    "optimization_method": "Benchmark-based resolution selection"
}

config_file = (
    "results/optimized_config.json"
)

with open(
    config_file,
    "w"
) as file:

    import json

    json.dump(
        config,
        file,
        indent=4
    )

# ==========================================
# DISPLAY
# ==========================================

print()

print("Benchmark results:")
print()

print(
    f"{'Image Size':<15}"
    f"{'FPS':<15}"
    f"{'Latency':<15}"
)

print("-" * 45)

for _, row in df.iterrows():

    print(
        f"{int(row['image_size']):<15}"
        f"{row['average_fps']:<15.2f}"
        f"{row['average_inference_ms']:<15.2f}"
    )

print()

print("=" * 60)
print("OPTIMIZATION RESULT")
print("=" * 60)

print()

print(
    f"Recommended image size: {best_size}"
)

print(
    f"Best FPS: {best_fps:.2f}"
)

print(
    f"Lowest latency: {best_latency:.2f} ms"
)

print()

print(
    "Optimization strategy:"
)

print(
    "Benchmark multiple inference resolutions "
    "and select the configuration with the highest FPS."
)

print()

print(
    f"Configuration saved to: {config_file}"
)

print("=" * 60)