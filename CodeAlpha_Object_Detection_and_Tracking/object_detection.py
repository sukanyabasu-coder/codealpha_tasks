import cv2
from ultralytics import YOLO
import numpy as np
from sort import Sort

# Load YOLO model
model = YOLO("yolov8s.pt")

# Run object detection on a sample image
results = model("https://ultralytics.com/images/bus.jpg")

# Show the result
results[0].show()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    
    # Read frame from webcam
    ret, frame = cap.read()
    
    # Run YOLO detection
    results = model(frame)
    
    # Draw detections on frame
    annotated_frame = results[0].plot()
    
    # Show output window
    cv2.imshow("YOLO Object Detection", annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()

import numpy as np
from sort import Sort

# Load YOLO model
model = YOLO("yolov8s.pt")

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    detections = np.empty((0, 5))

    # Process detections
    for result in results:

        boxes = result.boxes

        for box in boxes:

            # Coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            conf = float(box.conf[0])

            # Class name
            cls = int(box.cls[0])
            class_name = model.names[cls]

            # Lower confidence threshold
            if conf > 0.3:

                # Add detection for SORT
                current_detection = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_detection))

                # Draw object label
                cv2.putText(frame,
                            f"{class_name} {conf:.2f}",
                            (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2)

    # Update SORT tracker
    tracked_objects = tracker.update(detections)

    # Draw tracking IDs
    for obj in tracked_objects:

        x1, y1, x2, y2, track_id = obj

        x1, y1, x2, y2, track_id = map(int, [x1, y1, x2, y2, track_id])

        # Draw rectangle
        cv2.rectangle(frame,
                      (x1, y1),
                      (x2, y2),
                      (0, 255, 0),
                      2)

        # Display tracking ID
        cv2.putText(frame,
                    f"ID: {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

    # Show output
    cv2.imshow("YOLO + SORT Object Tracking", frame)

    # Quit on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
