Real-Time Face Detection & Unique Face Capture with Timestamp
A real-time face detection system using OpenCV, which captures and stores only unique faces with timestamps, preventing duplicate entries and optimizing memory usage.

📌 Features
✔ Real-time face detection using OpenCV
✔ Prevents duplicate captures using face similarity comparison
✔ Adds timestamps to saved face images for better tracking
✔ Skips unnecessary frames to optimize efficiency
✔ Includes failed attempts to document the learning process

📌 Tech Stack
Python
OpenCV (Face Detection)
NumPy (Image Processing)
Datetime (Timestamping)
📌 Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/face-detection-timestamp.git
cd face-detection-timestamp
2️⃣ Install Dependencies
pip install opencv-python numpy
3️⃣ Run the Final Working Script
python "Real time face detection with time stamp.py"
(Or open Face-detection.ipynb in Jupyter Notebook.)

📌 How It Works
Detects faces using OpenCV's Haar Cascade model.
Prevents duplicate captures by comparing detected faces.
Adds timestamps to saved face images for tracking.
Skips unnecessary frames to improve efficiency.
📌 Output
All unique detected faces are saved in the captured_faces/ directory with timestamps:

captured_faces/face_0_2024-02-12_14-35-20.jpg
captured_faces/face_1_2024-02-12_14-35-45.jpg
captured_faces/face_2_2024-02-12_14-36-01.jpg
📌 Repository Structure
📂 face-detection-timestamp
 ├── 📂 captured_faces          # Folder for stored face images
 ├── 📜 Real time face detection with time stamp.py  # Final working Python script
 ├── 📜 Failed codes.ipynb       # Notebook containing failed attempts & learning
 ├── 📜 README.md                # Project documentation
📌 Learning from Failed Attempts
This repository documents multiple failed attempts in Failed codes.ipynb, showcasing:

Different methods tested for face detection and duplicate prevention.
Challenges faced with varying face similarity models.
Final optimized solution that ensures unique face storage.
📌 Known Issues & Future Improvements
✅ Current Limitation:

Slight facial variations might cause a new face to be stored.
✅ Future Improvements:

Use deep learning models (dlib, face-recognition) for better accuracy.
Integrate with a database (SQLite, PostgreSQL) for persistent storage.
Add real-time face recognition to label known persons.
📌 Contributing
Feel free to fork this repo, submit pull requests, or suggest improvements! 🚀
