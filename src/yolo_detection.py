from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo11n.pt")

# Run object detection on a sample image
results = model("https://ultralytics.com/images/bus.jpg")

# Display the detection result
results[0].show()

print("YOLO detection completed successfully!")