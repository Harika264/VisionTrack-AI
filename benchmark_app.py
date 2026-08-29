import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="VisionTrack AI Benchmark",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ VisionTrack AI")
st.subheader("YOLO11n Performance Benchmark")

st.write(
    "Comparison of inference performance at different image sizes "
    "on the same CPU-based system."
)

benchmark_file = "results/model_benchmark.csv"

if not os.path.exists(benchmark_file):

    st.error(
        "Benchmark file not found."
    )

    st.info(
        "Run: python benchmark.py"
    )

    st.stop()

# Load benchmark
df = pd.read_csv(
    benchmark_file
)

# ==========================================
# SUMMARY
# ==========================================

best_fps_row = df.loc[
    df["average_fps"].idxmax()
]

best_latency_row = df.loc[
    df["average_inference_ms"].idxmin()
]

st.divider()

col1, col2, col3 = st.columns(3)

col1.metric(
    "🏆 Best FPS",
    f"{best_fps_row['average_fps']:.2f}"
)

col2.metric(
    "⚡ Best Image Size",
    int(best_fps_row["image_size"])
)

col3.metric(
    "🧠 Lowest Latency",
    f"{best_latency_row['average_inference_ms']:.2f} ms"
)

# ==========================================
# PERFORMANCE TABLE
# ==========================================

st.divider()

st.header("📊 Benchmark Results")

display_df = df.copy()

display_df.columns = [
    "Image Size",
    "Frames Tested",
    "Average FPS",
    "Inference Time (ms)",
    "Runtime (sec)"
]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# ==========================================
# FPS CHART
# ==========================================

st.divider()

st.header("⚡ FPS Comparison")

fps_chart = df.set_index(
    "image_size"
)["average_fps"]

st.bar_chart(
    fps_chart
)

# ==========================================
# LATENCY CHART
# ==========================================

st.divider()

st.header("🧠 Inference Latency Comparison")

latency_chart = df.set_index(
    "image_size"
)["average_inference_ms"]

st.bar_chart(
    latency_chart
)

# ==========================================
# INTERPRETATION
# ==========================================

st.divider()

st.header("🔬 Performance Interpretation")

best_size = int(
    best_fps_row["image_size"]
)

best_fps = float(
    best_fps_row["average_fps"]
)

best_latency = float(
    best_latency_row["average_inference_ms"]
)

st.success(
    f"🏆 Recommended configuration: "
    f"YOLO11n with {best_size}px inference size"
)

st.write(
    f"The benchmark achieved a maximum processing speed "
    f"of **{best_fps:.2f} FPS**."
)

st.write(
    f"The lowest measured inference latency was "
    f"**{best_latency:.2f} ms**."
)

st.write(
    "Smaller inference resolutions generally reduce "
    "computational cost and improve CPU inference speed, "
    "although very small resolutions can reduce detection "
    "accuracy for small or distant objects."
)

# ==========================================
# DOWNLOAD
# ==========================================

st.divider()

st.header("📥 Download Benchmark")

with open(
    benchmark_file,
    "rb"
) as file:

    st.download_button(
        label="📊 Download Benchmark CSV",
        data=file,
        file_name="visiontrack_model_benchmark.csv",
        mime="text/csv"
    )

# ==========================================
# PROJECT INFO
# ==========================================

st.divider()

st.header("💻 System Configuration")

st.write(
    "**Model:** YOLO11n"
)

st.write(
    "**Device:** CPU"
)

st.write(
    "**Benchmark Frames:** 100 per configuration"
)

st.write(
    "**Tested Image Sizes:** 640, 512, 416, 320"
)

st.caption(
    "VisionTrack AI — Computer Vision & Real-Time Video Analytics"
)