import cv2
import os

video_path = "input/test_video.mp4"

print("Current folder:")
print(os.getcwd())

print("\nVideo path:")
print(video_path)

print("\nFile exists:")
print(os.path.exists(video_path))

cap = cv2.VideoCapture(video_path)

print("\nVideo opened:")
print(cap.isOpened())

if cap.isOpened():
    print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FPS:", cap.get(cv2.CAP_PROP_FPS))
    print("Frame count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame = cap.read()

    print("First frame read:", ret)

    if ret:
        print("Video is readable by OpenCV!")

cap.release()