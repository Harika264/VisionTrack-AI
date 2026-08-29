import cv2
import os
import glob

# ==========================================
# VISIONTRACK AI
# Automatic Frame Extraction
# ==========================================

INPUT_DIR = "input"
OUTPUT_DIR = "results/evaluation_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find video files automatically
video_extensions = [
    "*.mp4",
    "*.avi",
    "*.mov",
    "*.mkv"
]

video_files = []

for extension in video_extensions:
    video_files.extend(
        glob.glob(
            os.path.join(INPUT_DIR, extension)
        )
    )

if not video_files:
    print("ERROR: No video found in input folder.")
    print()
    print("Make sure your video is inside:")
    print("VisionTrack-AI\\input")
    exit()

# Use the first video found
VIDEO_PATH = video_files[0]

print("================================")
print("VISIONTRACK AI")
print("FRAME EXTRACTION")
print("================================")

print(f"Video found: {VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print()
    print("ERROR: Video was found but could not be opened.")
    print(f"Path: {VIDEO_PATH}")
    exit()

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

duration = (
    total_frames / fps
    if fps > 0
    else 0
)

print(f"Total frames: {total_frames}")
print(f"Video FPS: {fps:.2f}")
print(f"Duration: {duration:.2f} seconds")

# ==========================================
# Extract approximately 50 frames
# ==========================================

number_of_frames = 50

interval = max(
    1,
    total_frames // number_of_frames
)

frame_number = 0
saved_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % interval == 0:

        filename = os.path.join(
            OUTPUT_DIR,
            f"frame_{saved_count:03d}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

        saved_count += 1

    frame_number += 1

cap.release()

print()
print("================================")
print("FRAME EXTRACTION COMPLETE")
print("================================")

print(f"Frames extracted: {saved_count}")
print(f"Saved to: {OUTPUT_DIR}")

print("================================")