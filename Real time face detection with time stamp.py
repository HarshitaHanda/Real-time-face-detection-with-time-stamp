#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import os
import numpy as np
from datetime import datetime  # Import for timestamp

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create a directory to store captured faces
face_dir = "captured_faces"
if not os.path.exists(face_dir):
    os.makedirs(face_dir)

# Open webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not access webcam.")
    exit()

face_count = len(os.listdir(face_dir))  # Start from last saved face count
last_captured_face = None  # Store the last captured face
frame_skip_count = 0  # Counter to skip frames before allowing a new capture
SKIP_THRESHOLD = 15  # How many frames to wait before checking for a new face

def is_duplicate_face(new_face):
    """Check if a new face is already stored by comparing against the last captured face."""
    global last_captured_face

    if last_captured_face is None:
        return False  # No previous face, so this is new

    # Convert both images to grayscale
    last_face_gray = cv2.cvtColor(last_captured_face, cv2.COLOR_BGR2GRAY)
    new_face_gray = cv2.cvtColor(new_face, cv2.COLOR_BGR2GRAY)

    # Resize both images to a fixed size for comparison
    last_face_gray = cv2.resize(last_face_gray, (100, 100))
    new_face_gray = cv2.resize(new_face_gray, (100, 100))

    # Compute the absolute difference
    diff = cv2.absdiff(last_face_gray, new_face_gray)
    score = np.mean(diff)  # Calculate mean difference

    return score < 10  # If difference is very low, consider it a duplicate

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to capture image. Exiting...")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces with increased `minSize` to filter out false detections
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80))

    for (x, y, w, h) in faces:
        # Ignore too small or too large faces (filter false detections)
        if w < 80 or h < 80:
            continue

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the face ROI in **color**
        face_color = frame[y:y+h, x:x+w]  # Keep the face in BGR (color)

        # Resize face for uniformity
        face_resized = cv2.resize(face_color, (200, 200))

        # Check if face is a duplicate using a **time delay mechanism**
        if frame_skip_count < SKIP_THRESHOLD:
            frame_skip_count += 1
            continue  # Skip saving the face for a few frames

        if not is_duplicate_face(face_resized):
            last_captured_face = face_resized.copy()  # Update the last captured face

            # Generate timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            face_filename = os.path.join(face_dir, f"face_{face_count}_{timestamp}.jpg")

            # Save the face image with timestamp
            cv2.imwrite(face_filename, face_resized)
            print(f"New unique face saved as: {face_filename}")

            face_count += 1
            frame_skip_count = 0  # Reset frame counter
        else:
            print("Same face detected. Skipping...")

    # Show the frame with detection
    cv2.imshow("Face Detection - Press 'q' to Exit", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()

