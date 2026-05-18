# CodeAlpha_Object_Detection_and_Tracking
A real-time computer vision project that performs object detection and object tracking using YOLOv8 and the SORT tracking algorithm. The system detects multiple objects through a webcam, assigns unique tracking IDs, and displays bounding boxes with confidence scores in real time.

# Real-Time Object Detection and Tracking using YOLOv8 and SORT

## Overview
This project performs real-time object detection and tracking using a webcam.  
It uses the YOLOv8 model for object detection and the SORT algorithm for assigning unique tracking IDs to detected objects.

The system can:
- Detect multiple real-world objects
- Track moving objects in real time
- Assign unique IDs to objects
- Display confidence scores and labels

---

## Technologies Used
- Python
- OpenCV
- YOLOv8
- SORT Tracking Algorithm
- NumPy

---

## Features
- Real-time webcam object detection
- Object tracking with unique IDs
- Bounding boxes around detected objects
- Confidence score display
- Multiple object handling
- Stable tracking improvements using SORT parameters

---

## Project Structure

```text
object_detection.py
sort.py
README.md
```

---

## Installation

Install the required libraries:

```python
pip install ultralytics
pip install opencv-python
pip install numpy
pip install filterpy
pip install scikit-image
```

---

## How to Run the Project

Run the Python file using CMD or Anaconda Prompt:

```python
python object_detection.py
```

Press:
```text
q
```

to close the webcam window.

---

## Working Process

1. Webcam captures live video frames
2. YOLOv8 detects objects in each frame
3. SORT tracker assigns unique IDs
4. Bounding boxes and IDs are displayed in real time

---

## Sample Detected Objects
- Person
- Bottle
- Phone
- Laptop
- Chair
- Book
- Cup

---

## Challenges Faced
- Webcam instability in Jupyter Notebook
- Python 3.13 compatibility issues with SORT
- Object flickering during low-confidence detection
- Detection inaccuracies for small objects

These issues were improved using:
- Lower confidence threshold
- Better lighting
- Stable SORT parameters
- Running the project through CMD

---

##**Output**

Sample ouput reference video is attached to have a quick look!

---

## Future Improvements
- DeepSORT integration
- Custom-trained YOLO model
- GPU acceleration
- Video file input support
- Object counting system

---

## Conclusion
This project demonstrates the implementation of real-time object detection and object tracking using YOLOv8 and SORT. It provides practical exposure to computer vision concepts such as detection, tracking, bounding boxes, confidence scores, and real-time webcam processing.

---

## Author
Sukanya
CodeAlpha Artificial Intelligence Internship
