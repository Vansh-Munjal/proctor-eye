# 👁️ AI-Based Gaze and Head Pose Distraction Monitoring System

An AI-powered web application designed to monitor candidate behavior during online interviews or proctored exams by tracking **gaze direction** and **head movement** using a webcam. The system detects distractions, triggers alerts, and generates a report of inattentiveness.

---

## 📌 Features

- 🔍 Real-time **gaze direction detection** (LEFT, CENTER, RIGHT)
- 👤 **Head pose estimation** using MediaPipe and facial landmarks
- ⏱️ Distraction timer and alert system
- ⚠️ Visual & audio alerts for cheating/distraction detection
- 📊 Distraction **summary report** generation
- 🌐 Web-based frontend connected to Flask backend
- ✅ Works alongside Zoom, Google Meet, MS Teams, etc.

---

## 🛠️ Tech Stack

| Area       | Tools/Libraries                    |
|------------|------------------------------------|
| Frontend   | HTML, CSS, JavaScript              |
| Backend    | Flask (Python)                     |
| Gaze Model | TensorFlow / Keras (CNN Model)     |
| Head Pose  | MediaPipe, OpenCV                  |
| Alerts     | Pygame (sound), HTML overlay       |
| Deployment | (Optional) Render, Vercel, etc.    |

---

## 📂 Folder Structure

