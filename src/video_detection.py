import cv2
import time
import csv
import os
from ultralytics import YOLO

# ==========================================
# VISIONTRACK AI
# Detection + Tracking + Trajectory + CSV
# ==========================================

# Load YOLO model
model = YOLO("yolo11n.pt")

# Video input
video_path = "input/test_video.mp4"

# Results directory
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# CSV output
csv_path = os.path.join(
    results_dir,
    "analytics.csv"
)

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

print("================================")
print("VISIONTRACK AI")
print("================================")
print("Video opened successfully!")
print("Starting video analytics...")
print("Press Q to stop.")

# ==========================================
# Tracking data
# ==========================================

tracked_objects = {
    "person": set(),
    "car": set(),
    "truck": set(),
    "bus": set(),
    "motorcycle": set(),
    "bicycle": set()
}

# Trajectories
trajectories = {}

MAX_TRAJECTORY_POINTS = 30

# Previous positions
previous_positions = {}

# Counted objects
counted_objects = set()

# Entry / Exit
vehicles_in = 0
vehicles_out = 0

# Counting line
LINE_Y = 400

# FPS
previous_time = time.time()
fps = 0

# FPS statistics
fps_values = []

# ==========================================
# Main loop
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    # ======================================
    # YOLO tracking
    # ======================================

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    result = results[0]

    # Draw detections
    annotated_frame = result.plot()

    # ======================================
    # Counting line
    # ======================================

    cv2.line(
        annotated_frame,
        (0, LINE_Y),
        (annotated_frame.shape[1], LINE_Y),
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

    # ======================================
    # Process tracked objects
    # ======================================

    if result.boxes is not None:

        boxes = result.boxes

        if boxes.id is not None:

            tracking_ids = boxes.id.int().tolist()
            class_ids = boxes.cls.int().tolist()
            coordinates = boxes.xyxy.tolist()

            for track_id, class_id, box in zip(
                tracking_ids,
                class_ids,
                coordinates
            ):

                class_name = model.names[class_id]

                # ==================================
                # Class tracking
                # ==================================

                if class_name in tracked_objects:
                    tracked_objects[class_name].add(track_id)

                # ==================================
                # Bounding box center
                # ==================================

                x1, y1, x2, y2 = box

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # ==================================
                # Draw center
                # ==================================

                cv2.circle(
                    annotated_frame,
                    (center_x, center_y),
                    5,
                    (0, 255, 0),
                    -1
                )

                # ==================================
                # Trajectory
                # ==================================

                if track_id not in trajectories:
                    trajectories[track_id] = []

                trajectories[track_id].append(
                    (center_x, center_y)
                )

                if len(trajectories[track_id]) > MAX_TRAJECTORY_POINTS:
                    trajectories[track_id].pop(0)

                points = trajectories[track_id]

                for i in range(1, len(points)):

                    cv2.line(
                        annotated_frame,
                        points[i - 1],
                        points[i],
                        (255, 0, 0),
                        2
                    )

                # ==================================
                # Entry / Exit
                # ==================================

                if track_id in previous_positions:

                    previous_y = previous_positions[track_id]

                    # Moving downward
                    if (
                        previous_y < LINE_Y
                        and center_y >= LINE_Y
                        and track_id not in counted_objects
                    ):

                        if class_name in [
                            "car",
                            "truck",
                            "bus",
                            "motorcycle"
                        ]:

                            vehicles_in += 1
                            counted_objects.add(track_id)

                            print(
                                f"Vehicle IN: "
                                f"{class_name} "
                                f"ID={track_id}"
                            )

                    # Moving upward
                    elif (
                        previous_y > LINE_Y
                        and center_y <= LINE_Y
                        and track_id not in counted_objects
                    ):

                        if class_name in [
                            "car",
                            "truck",
                            "bus",
                            "motorcycle"
                        ]:

                            vehicles_out += 1
                            counted_objects.add(track_id)

                            print(
                                f"Vehicle OUT: "
                                f"{class_name} "
                                f"ID={track_id}"
                            )

                previous_positions[track_id] = center_y

    # ======================================
    # FPS
    # ======================================

    current_time = time.time()

    elapsed = current_time - previous_time

    if elapsed > 0:

        fps = 1 / elapsed
        fps_values.append(fps)

    previous_time = current_time

    # ======================================
    # Analytics panel
    # ======================================

    cv2.rectangle(
        annotated_frame,
        (10, 10),
        (500, 280),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        annotated_frame,
        "VISIONTRACK AI",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Persons: {len(tracked_objects['person'])}",
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Cars: {len(tracked_objects['car'])}",
        (25, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Trucks: {len(tracked_objects['truck'])}",
        (25, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Vehicles IN: {vehicles_in}",
        (260, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Vehicles OUT: {vehicles_out}",
        (260, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (260, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        "Tracking: ACTIVE",
        (25, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        "Trajectory: ACTIVE",
        (25, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # ======================================
    # Display
    # ======================================

    cv2.imshow(
        "VisionTrack AI - Video Analytics",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# Cleanup
# ==========================================

cap.release()
cv2.destroyAllWindows()

# ==========================================
# Calculate average FPS
# ==========================================

if fps_values:
    average_fps = sum(fps_values) / len(fps_values)
else:
    average_fps = 0

# ==========================================
# Save CSV report
# ==========================================

with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Metric",
        "Value"
    ])

    writer.writerow([
        "Persons Tracked",
        len(tracked_objects["person"])
    ])

    writer.writerow([
        "Cars Tracked",
        len(tracked_objects["car"])
    ])

    writer.writerow([
        "Trucks Tracked",
        len(tracked_objects["truck"])
    ])

    writer.writerow([
        "Buses Tracked",
        len(tracked_objects["bus"])
    ])

    writer.writerow([
        "Motorcycles Tracked",
        len(tracked_objects["motorcycle"])
    ])

    writer.writerow([
        "Bicycles Tracked",
        len(tracked_objects["bicycle"])
    ])

    writer.writerow([
        "Vehicles IN",
        vehicles_in
    ])

    writer.writerow([
        "Vehicles OUT",
        vehicles_out
    ])

    writer.writerow([
        "Average FPS",
        round(average_fps, 2)
    ])

# ==========================================
# Final report
# ==========================================

print()
print("================================")
print("VISIONTRACK AI FINAL REPORT")
print("================================")

print(
    f"Persons tracked: "
    f"{len(tracked_objects['person'])}"
)

print(
    f"Cars tracked: "
    f"{len(tracked_objects['car'])}"
)

print(
    f"Trucks tracked: "
    f"{len(tracked_objects['truck'])}"
)

print(
    f"Vehicles IN: "
    f"{vehicles_in}"
)

print(
    f"Vehicles OUT: "
    f"{vehicles_out}"
)

print(
    f"Average FPS: "
    f"{average_fps:.2f}"
)

print(
    f"CSV report saved: "
    f"{csv_path}"
)

print("================================")