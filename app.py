import streamlit as st
import cv2
import tempfile
import os
import time
import json
import csv
import torch
import pandas as pd
from ultralytics import YOLO

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="VisionTrack AI",
    page_icon="🎯",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================

st.title("🎯 VisionTrack AI")
st.subheader("Intelligent Computer Vision & Real-Time Video Analytics")

st.write(
    "YOLO-based object detection, multi-object tracking, "
    "trajectory analysis, vehicle counting and performance benchmarking."
)

# ==========================================
# DEVICE
# ==========================================

device = "GPU" if torch.cuda.is_available() else "CPU"

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


model = load_model()

# ==========================================
# FILE PATHS
# ==========================================

benchmark_file = "results/model_benchmark.csv"
optimization_file = "results/optimized_config.json"
evaluation_file = "results/evaluation_metrics.csv"

# ==========================================
# LOAD OPTIMIZATION
# ==========================================

recommended_size = 320
optimization_data = {}

if os.path.exists(optimization_file):

    try:

        with open(
            optimization_file,
            "r"
        ) as file:

            optimization_data = json.load(file)

        recommended_size = int(
            optimization_data[
                "optimized_image_size"
            ]
        )

    except Exception:
        recommended_size = 320

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("⚙️ Model Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    0.1,
    1.0,
    0.5,
    0.05
)

use_optimized_size = st.sidebar.checkbox(
    "🚀 Use Optimized Resolution",
    True
)

if use_optimized_size:

    image_size = recommended_size

    st.sidebar.success(
        f"Optimized Size: {image_size}px"
    )

else:

    image_size = st.sidebar.selectbox(
        "Inference Image Size",
        [320, 416, 512, 640],
        index=1
    )

show_trajectory = st.sidebar.checkbox(
    "Show Trajectories",
    True
)

st.sidebar.divider()

st.sidebar.write("### System")
st.sidebar.write("🤖 Model: YOLO11n")
st.sidebar.write(f"💻 Device: {device}")

# ==========================================
# OPTIMIZATION SUMMARY
# ==========================================

if os.path.exists(optimization_file):

    st.sidebar.divider()

    st.sidebar.write("### 🏆 Optimization")

    st.sidebar.write(
        f"Recommended: **{recommended_size}px**"
    )

    if optimization_data:

        st.sidebar.write(
            f"Best FPS: "
            f"**{optimization_data.get('best_fps', 0):.2f}**"
        )

        st.sidebar.write(
            f"Lowest latency: "
            f"**{optimization_data.get('lowest_latency_ms', 0):.2f} ms**"
        )

# ==========================================
# VIDEO UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📁 Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)

# ==========================================
# MAIN VIDEO ANALYSIS
# ==========================================

if uploaded_file is not None:

    st.success(
        f"Video uploaded: {uploaded_file.name}"
    )

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_file.write(
        uploaded_file.read()
    )

    temp_file.close()

    video_path = temp_file.name

    if st.button(
        "🚀 Start Video Analysis",
        type="primary"
    ):

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            st.error(
                "Could not open the uploaded video."
            )

        else:

            # ==================================
            # VIDEO INFORMATION
            # ==================================

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            video_fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            video_width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            video_height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            # ==================================
            # TRACKING DATA
            # ==================================

            tracked_objects = {
                "person": set(),
                "car": set(),
                "truck": set(),
                "bus": set(),
                "motorcycle": set(),
                "bicycle": set()
            }

            trajectories = {}
            previous_positions = {}
            counted_objects = set()

            vehicles_in = 0
            vehicles_out = 0

            LINE_Y = int(
                video_height * 0.55
            )

            MAX_POINTS = 30

            # ==================================
            # PERFORMANCE
            # ==================================

            frame_count = 0
            total_inference_time = 0
            total_processing_time = 0

            start_time = time.perf_counter()

            # ==================================
            # LIVE DASHBOARD
            # ==================================

            st.subheader(
                "📊 Live Performance"
            )

            col1, col2, col3, col4 = st.columns(4)

            person_metric = col1.empty()
            car_metric = col2.empty()
            truck_metric = col3.empty()
            fps_metric = col4.empty()

            col5, col6, col7, col8 = st.columns(4)

            inference_metric = col5.empty()
            frames_metric = col6.empty()
            in_metric = col7.empty()
            out_metric = col8.empty()

            st.divider()

            video_placeholder = st.empty()

            # ==================================
            # PROCESS VIDEO
            # ==================================

            while True:

                frame_start = time.perf_counter()

                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # ==================================
                # YOLO TRACKING
                # ==================================

                inference_start = time.perf_counter()

                results = model.track(
                    frame,
                    persist=True,
                    conf=confidence,
                    imgsz=image_size,
                    device=(
                        0
                        if torch.cuda.is_available()
                        else "cpu"
                    ),
                    verbose=False
                )

                inference_time = (
                    time.perf_counter()
                    - inference_start
                )

                total_inference_time += (
                    inference_time
                )

                result = results[0]

                annotated_frame = (
                    result.plot()
                )

                # ==================================
                # COUNTING LINE
                # ==================================

                cv2.line(
                    annotated_frame,
                    (0, LINE_Y),
                    (video_width, LINE_Y),
                    (0, 255, 255),
                    3
                )

                cv2.putText(
                    annotated_frame,
                    "COUNTING LINE",
                    (20, LINE_Y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                # ==================================
                # TRACK OBJECTS
                # ==================================

                if result.boxes is not None:

                    boxes = result.boxes

                    if boxes.id is not None:

                        ids = boxes.id.int().tolist()
                        classes = boxes.cls.int().tolist()
                        coordinates = boxes.xyxy.tolist()

                        for (
                            track_id,
                            class_id,
                            box
                        ) in zip(
                            ids,
                            classes,
                            coordinates
                        ):

                            class_name = model.names[
                                class_id
                            ]

                            # --------------------------
                            # OBJECT COUNT
                            # --------------------------

                            if class_name in tracked_objects:

                                tracked_objects[
                                    class_name
                                ].add(track_id)

                            # --------------------------
                            # CENTER
                            # --------------------------

                            x1, y1, x2, y2 = box

                            center_x = int(
                                (x1 + x2) / 2
                            )

                            center_y = int(
                                (y1 + y2) / 2
                            )

                            # --------------------------
                            # TRAJECTORY
                            # --------------------------

                            if show_trajectory:

                                if track_id not in trajectories:

                                    trajectories[
                                        track_id
                                    ] = []

                                trajectories[
                                    track_id
                                ].append(
                                    (
                                        center_x,
                                        center_y
                                    )
                                )

                                if len(
                                    trajectories[
                                        track_id
                                    ]
                                ) > MAX_POINTS:

                                    trajectories[
                                        track_id
                                    ].pop(0)

                                points = trajectories[
                                    track_id
                                ]

                                for i in range(
                                    1,
                                    len(points)
                                ):

                                    cv2.line(
                                        annotated_frame,
                                        points[i - 1],
                                        points[i],
                                        (255, 0, 0),
                                        2
                                    )

                            # --------------------------
                            # IN / OUT
                            # --------------------------

                            if track_id in previous_positions:

                                previous_y = (
                                    previous_positions[
                                        track_id
                                    ]
                                )

                                vehicle_class = (
                                    class_name in [
                                        "car",
                                        "truck",
                                        "bus",
                                        "motorcycle"
                                    ]
                                )

                                if (
                                    previous_y < LINE_Y
                                    and center_y >= LINE_Y
                                    and track_id not in counted_objects
                                    and vehicle_class
                                ):

                                    vehicles_in += 1

                                    counted_objects.add(
                                        track_id
                                    )

                                elif (
                                    previous_y > LINE_Y
                                    and center_y <= LINE_Y
                                    and track_id not in counted_objects
                                    and vehicle_class
                                ):

                                    vehicles_out += 1

                                    counted_objects.add(
                                        track_id
                                    )

                            previous_positions[
                                track_id
                            ] = center_y

                # ==================================
                # PERFORMANCE
                # ==================================

                frame_end = time.perf_counter()

                processing_time = (
                    frame_end -
                    frame_start
                )

                total_processing_time += (
                    processing_time
                )

                average_inference_ms = (
                    total_inference_time /
                    frame_count
                ) * 1000

                average_fps = (
                    frame_count /
                    total_processing_time
                    if total_processing_time > 0
                    else 0
                )

                # ==================================
                # LIVE METRICS
                # ==================================

                person_metric.metric(
                    "👤 Persons",
                    len(
                        tracked_objects[
                            "person"
                        ]
                    )
                )

                car_metric.metric(
                    "🚗 Cars",
                    len(
                        tracked_objects[
                            "car"
                        ]
                    )
                )

                truck_metric.metric(
                    "🚚 Trucks",
                    len(
                        tracked_objects[
                            "truck"
                        ]
                    )
                )

                fps_metric.metric(
                    "⚡ Processing FPS",
                    f"{average_fps:.2f}"
                )

                inference_metric.metric(
                    "🧠 Avg Inference",
                    f"{average_inference_ms:.2f} ms"
                )

                frames_metric.metric(
                    "🎞️ Frames",
                    frame_count
                )

                in_metric.metric(
                    "↘ Vehicles IN",
                    vehicles_in
                )

                out_metric.metric(
                    "↗ Vehicles OUT",
                    vehicles_out
                )

                video_placeholder.image(
                    cv2.cvtColor(
                        annotated_frame,
                        cv2.COLOR_BGR2RGB
                    ),
                    channels="RGB"
                )

            # ==================================
            # FINISH
            # ==================================

            cap.release()

            total_runtime = (
                time.perf_counter()
                - start_time
            )

            average_fps = (
                frame_count /
                total_runtime
                if total_runtime > 0
                else 0
            )

            average_inference_ms = (
                total_inference_time /
                frame_count
            ) * 1000 if frame_count > 0 else 0

            # ==================================
            # FINAL COUNTS
            # ==================================

            persons = len(
                tracked_objects["person"]
            )

            cars = len(
                tracked_objects["car"]
            )

            trucks = len(
                tracked_objects["truck"]
            )

            buses = len(
                tracked_objects["bus"]
            )

            motorcycles = len(
                tracked_objects["motorcycle"]
            )

            bicycles = len(
                tracked_objects["bicycle"]
            )

            # ==================================
            # REPORT
            # ==================================

            report = {
                "project": "VisionTrack AI",
                "video": uploaded_file.name,
                "model": "YOLO11n",
                "device": device,
                "confidence": confidence,
                "image_size": image_size,
                "optimized_configuration": use_optimized_size,
                "recommended_image_size": recommended_size,
                "video_resolution":
                    f"{video_width}x{video_height}",
                "original_video_fps":
                    round(video_fps, 2),
                "frames_processed":
                    frame_count,
                "average_fps":
                    round(average_fps, 2),
                "average_inference_ms":
                    round(
                        average_inference_ms,
                        2
                    ),
                "total_runtime_seconds":
                    round(
                        total_runtime,
                        2
                    ),
                "persons_tracked":
                    persons,
                "cars_tracked":
                    cars,
                "trucks_tracked":
                    trucks,
                "buses_tracked":
                    buses,
                "motorcycles_tracked":
                    motorcycles,
                "bicycles_tracked":
                    bicycles,
                "vehicles_in":
                    vehicles_in,
                "vehicles_out":
                    vehicles_out,
                "trajectory_tracking":
                    (
                        "ACTIVE"
                        if show_trajectory
                        else "DISABLED"
                    ),
                "object_tracking":
                    "ACTIVE"
            }

            os.makedirs(
                "results",
                exist_ok=True
            )

            # ==================================
            # JSON
            # ==================================

            json_path = (
                "results/visiontrack_report.json"
            )

            with open(
                json_path,
                "w"
            ) as file:

                json.dump(
                    report,
                    file,
                    indent=4
                )

            # ==================================
            # CSV
            # ==================================

            csv_path = (
                "results/visiontrack_report.csv"
            )

            with open(
                csv_path,
                "w",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow(
                    [
                        "Metric",
                        "Value"
                    ]
                )

                for key, value in report.items():

                    writer.writerow(
                        [
                            key,
                            value
                        ]
                    )

            # ==================================
            # SUCCESS
            # ==================================

            st.success(
                "✅ Video analysis completed successfully!"
            )

            # ==================================
            # FINAL PERFORMANCE
            # ==================================

            st.divider()

            st.header(
                "📈 Final Performance Report"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "🎞️ Frames Processed",
                frame_count
            )

            col2.metric(
                "⚡ Average FPS",
                f"{average_fps:.2f}"
            )

            col3.metric(
                "🧠 Inference Time",
                f"{average_inference_ms:.2f} ms"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "👤 Persons",
                persons
            )

            col2.metric(
                "🚗 Cars",
                cars
            )

            col3.metric(
                "🚚 Trucks",
                trucks
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "↘ Vehicles IN",
                vehicles_in
            )

            col2.metric(
                "↗ Vehicles OUT",
                vehicles_out
            )

            col3.metric(
                "⏱️ Runtime",
                f"{total_runtime:.2f} sec"
            )

            # ==================================
            # SYSTEM CONFIG
            # ==================================

            st.divider()

            st.header(
                "💻 System & Model Configuration"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    "**Model:** YOLO11n"
                )

                st.write(
                    f"**Device:** {device}"
                )

                st.write(
                    f"**Inference Size:** "
                    f"{image_size}"
                )

                st.write(
                    f"**Detection Confidence:** "
                    f"{confidence:.2f}"
                )

            with col2:

                st.write(
                    f"**Input Resolution:** "
                    f"{video_width} × "
                    f"{video_height}"
                )

                st.write(
                    f"**Original Video FPS:** "
                    f"{video_fps:.2f}"
                )

                st.write(
                    f"**Optimized Size:** "
                    f"{recommended_size}"
                )

                st.write(
                    "**Trajectory Tracking:** "
                    f"{'ACTIVE' if show_trajectory else 'DISABLED'}"
                )

            # ==================================
            # BENCHMARK
            # ==================================

            if os.path.exists(
                benchmark_file
            ):

                st.divider()

                st.header(
                    "⚡ Model Performance Benchmark"
                )

                benchmark_df = pd.read_csv(
                    benchmark_file
                )

                best_fps_row = benchmark_df.loc[
                    benchmark_df[
                        "average_fps"
                    ].idxmax()
                ]

                best_latency_row = benchmark_df.loc[
                    benchmark_df[
                        "average_inference_ms"
                    ].idxmin()
                ]

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "🏆 Best FPS",
                    f"{best_fps_row['average_fps']:.2f}"
                )

                col2.metric(
                    "🎯 Best Resolution",
                    f"{int(best_fps_row['image_size'])} px"
                )

                col3.metric(
                    "🧠 Lowest Latency",
                    f"{best_latency_row['average_inference_ms']:.2f} ms"
                )

                display_df = benchmark_df.copy()

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

                st.subheader(
                    "📊 FPS Comparison"
                )

                st.bar_chart(
                    benchmark_df.set_index(
                        "image_size"
                    )["average_fps"]
                )

                st.subheader(
                    "🧠 Inference Latency Comparison"
                )

                st.bar_chart(
                    benchmark_df.set_index(
                        "image_size"
                    )["average_inference_ms"]
                )

                st.info(
                    f"🏆 The benchmark identified "
                    f"{int(best_fps_row['image_size'])}px "
                    f"as the fastest tested resolution "
                    f"with {best_fps_row['average_fps']:.2f} FPS."
                )

                st.caption(
                    "Benchmark FPS represents controlled "
                    "model performance and should not be "
                    "treated as the full Streamlit pipeline FPS."
                )

            # ==================================
            # MODEL EVALUATION
            # ==================================

            if os.path.exists(
                evaluation_file
            ):

                st.divider()

                st.header(
                    "🔬 Model Evaluation"
                )

                evaluation_df = pd.read_csv(
                    evaluation_file
                )

                evaluation_data = {}

                for _, row in evaluation_df.iterrows():

                    evaluation_data[
                        str(row["Metric"])
                    ] = row["Value"]

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "🎞️ Frames Evaluated",
                    evaluation_data.get(
                        "Frames Evaluated",
                        0
                    )
                )

                col2.metric(
                    "⚡ Inference FPS",
                    evaluation_data.get(
                        "Inference FPS",
                        0
                    )
                )

                col3.metric(
                    "🧠 Avg Confidence",
                    f"{float(evaluation_data.get('Average Detection Confidence', 0)) * 100:.1f}%"
                )

                col4.metric(
                    "📦 Objects Detected",
                    evaluation_data.get(
                        "Total Objects Detected",
                        0
                    )
                )

                st.subheader(
                    "Object Detection Distribution"
                )

                class_data = {

                    "Persons":
                        float(
                            evaluation_data.get(
                                "Persons Detected",
                                0
                            )
                        ),

                    "Cars":
                        float(
                            evaluation_data.get(
                                "Cars Detected",
                                0
                            )
                        ),

                    "Trucks":
                        float(
                            evaluation_data.get(
                                "Trucks Detected",
                                0
                            )
                        ),

                    "Buses":
                        float(
                            evaluation_data.get(
                                "Buses Detected",
                                0
                            )
                        ),

                    "Motorcycles":
                        float(
                            evaluation_data.get(
                                "Motorcycles Detected",
                                0
                            )
                        ),

                    "Bicycles":
                        float(
                            evaluation_data.get(
                                "Bicycles Detected",
                                0
                            )
                        )
                }

                class_df = pd.DataFrame(
                    {
                        "Object Class":
                            list(class_data.keys()),

                        "Detections":
                            list(class_data.values())
                    }
                )

                st.bar_chart(
                    class_df.set_index(
                        "Object Class"
                    )
                )

                st.subheader(
                    "Evaluation Metrics"
                )

                st.dataframe(
                    evaluation_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.info(
                    "The evaluation confidence shown above "
                    "is the average confidence of model "
                    "detections. It is not an accuracy, "
                    "precision, recall or mAP score."
                )

                with open(
                    evaluation_file,
                    "rb"
                ) as file:

                    st.download_button(
                        "📥 Download Evaluation Results",
                        file,
                        file_name="evaluation_metrics.csv",
                        mime="text/csv"
                    )

            # ==================================
            # DOWNLOAD REPORTS
            # ==================================

            st.divider()

            st.header(
                "📥 Download Analytics Reports"
            )

            with open(
                json_path,
                "r"
            ) as file:

                json_data = file.read()

            with open(
                csv_path,
                "r"
            ) as file:

                csv_data = file.read()

            col1, col2 = st.columns(2)

            col1.download_button(
                "📄 Download JSON Report",
                json_data,
                file_name="visiontrack_report.json",
                mime="application/json"
            )

            col2.download_button(
                "📊 Download CSV Report",
                csv_data,
                file_name="visiontrack_report.csv",
                mime="text/csv"
            )

            # ==================================
            # PERFORMANCE INTERPRETATION
            # ==================================

            st.divider()

            st.header(
                "🔬 Performance Interpretation"
            )

            if average_fps >= 20:

                st.success(
                    "🚀 Excellent processing performance "
                    "for the current configuration."
                )

            elif average_fps >= 15:

                st.success(
                    "✅ Good processing performance."
                )

            elif average_fps >= 8:

                st.warning(
                    "⚠️ Moderate processing performance."
                )

            else:

                st.info(
                    "ℹ️ The complete Streamlit pipeline "
                    "is CPU-limited. The benchmark has "
                    "already identified 320px as the "
                    "fastest tested inference resolution. "
                    "Further improvement would require "
                    "reducing processing/visualization "
                    "overhead or using hardware acceleration."
                )

            st.write(
                f"Current analysis resolution: "
                f"**{image_size}px**"
            )

            st.write(
                f"Full application FPS: "
                f"**{average_fps:.2f} FPS**"
            )

            st.write(
                f"Average YOLO inference latency: "
                f"**{average_inference_ms:.2f} ms**"
            )

    try:

        os.unlink(video_path)

    except:

        pass

else:

    st.info(
        "👆 Upload a video to start VisionTrack AI."
    )

    # ==========================================
    # HOME PAGE
    # ==========================================

    if os.path.exists(
        benchmark_file
    ):

        st.divider()

        st.header(
            "⚡ Current Model Benchmark"
        )

        benchmark_df = pd.read_csv(
            benchmark_file
        )

        best_row = benchmark_df.loc[
            benchmark_df[
                "average_fps"
            ].idxmax()
        ]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "🏆 Best FPS",
            f"{best_row['average_fps']:.2f}"
        )

        col2.metric(
            "🎯 Recommended Size",
            f"{int(best_row['image_size'])} px"
        )

        col3.metric(
            "🧠 Best Latency",
            f"{benchmark_df['average_inference_ms'].min():.2f} ms"
        )

        st.bar_chart(
            benchmark_df.set_index(
                "image_size"
            )["average_fps"]
        )

    # ==========================================
    # EVALUATION SUMMARY
    # ==========================================

    if os.path.exists(
        evaluation_file
    ):

        st.divider()

        st.header(
            "🔬 Latest Model Evaluation"
        )

        evaluation_df = pd.read_csv(
            evaluation_file
        )

        evaluation_data = {}

        for _, row in evaluation_df.iterrows():

            evaluation_data[
                str(row["Metric"])
            ] = row["Value"]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Avg Confidence",
            f"{float(evaluation_data.get('Average Detection Confidence', 0)) * 100:.1f}%"
        )

        col2.metric(
            "Inference FPS",
            evaluation_data.get(
                "Inference FPS",
                0
            )
        )

        col3.metric(
            "Objects Detected",
            evaluation_data.get(
                "Total Objects Detected",
                0
            )
        )