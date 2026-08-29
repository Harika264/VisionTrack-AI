import cv2
import os
import glob
import json

FRAME_DIR = "results/evaluation_frames"
OUTPUT_FILE = "results/ground_truth.json"

frames = sorted(
    glob.glob(os.path.join(FRAME_DIR, "*.jpg"))
)

if not frames:
    print("ERROR: No frames found.")
    exit()

ground_truth = {}

print("=" * 50)
print("VISIONTRACK AI - GROUND TRUTH LABELING")
print("=" * 50)
print()
print(f"Total frames available: {len(frames)}")
print()
print("For every frame:")
print("Count the visible PERSONS, CARS and TRUCKS.")
print()
print("You will enter them like:")
print("2 5 1")
print("= 2 persons, 5 cars, 1 truck")
print()
print("Type S to skip a frame.")
print("Type Q to stop and save.")
print("=" * 50)

for index, frame_path in enumerate(frames):

    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"Could not read {frame_path}")
        continue

    # Resize image for easier viewing
    height, width = frame.shape[:2]

    max_width = 1000
    max_height = 650

    scale = min(
        max_width / width,
        max_height / height,
        1.0
    )

    if scale < 1:
        frame = cv2.resize(
            frame,
            (
                int(width * scale),
                int(height * scale)
            )
        )

    # Add frame number
    cv2.putText(
        frame,
        f"Frame {index + 1} / {len(frames)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    window_name = "VisionTrack AI - Label Frame"

    cv2.imshow(window_name, frame)

    print()
    print("-" * 50)
    print(f"FRAME {index + 1} OF {len(frames)}")
    print(f"Image: {os.path.basename(frame_path)}")
    print("-" * 50)

    print()
    print("Look at the image window.")
    print("Close the image window using X.")
    print()

    # Keep window alive until user closes it
    while True:

        cv2.waitKey(100)

        try:
            visible = cv2.getWindowProperty(
                window_name,
                cv2.WND_PROP_VISIBLE
            )

            if visible < 1:
                break

        except:
            break

    cv2.destroyAllWindows()

    # Get label from terminal
    while True:

        answer = input(
            "Enter Persons Cars Trucks "
            "(example: 2 5 1): "
        ).strip()

        if answer.lower() == "q":
            cv2.destroyAllWindows()
            break

        if answer.lower() == "s":
            print("Frame skipped.")
            break

        parts = answer.split()

        if len(parts) != 3:
            print(
                "ERROR: Enter exactly 3 numbers."
            )
            print(
                "Example: 2 5 1"
            )
            continue

        try:
            persons = int(parts[0])
            cars = int(parts[1])
            trucks = int(parts[2])

            if persons < 0 or cars < 0 or trucks < 0:
                raise ValueError

        except ValueError:
            print(
                "ERROR: Use numbers greater than or equal to 0."
            )
            continue

        ground_truth[
            os.path.basename(frame_path)
        ] = {
            "person": persons,
            "car": cars,
            "truck": trucks
        }

        print()
        print("✓ Saved:")
        print(f"  Persons: {persons}")
        print(f"  Cars:    {cars}")
        print(f"  Trucks:  {trucks}")

        break

    if answer.lower() == "q":
        break

# Save results
os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w"
) as file:

    json.dump(
        ground_truth,
        file,
        indent=4
    )

cv2.destroyAllWindows()

print()
print("=" * 50)
print("GROUND TRUTH LABELING COMPLETE")
print("=" * 50)
print(f"Frames labeled: {len(ground_truth)}")
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 50)