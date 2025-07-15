from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import cv2
import numpy as np
import base64
import time
import pygame
import mediapipe as mp
import os, json
from datetime import datetime
from tensorflow.keras.models import load_model
from eye_tracker import crop_eye, preprocess_eye
from head_pose import get_head_pose

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Change for production

# Load model and labels
model = load_model("models/gaze_cnn.h5")
labels = ["LEFT", "CENTER", "RIGHT"]

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Sound alert
pygame.mixer.init()
pygame.mixer.music.load("assets/alert.mp3")

# Global alert state
start_distract_time = None
alert_triggered = False
cheating_displayed = False

# ------------------ Utility ------------------

def play_alert():
    pygame.mixer.music.play()

def is_distracted(gaze, head):
    return gaze != "CENTER" or head in ["LEFT", "RIGHT"]

# ------------------ Routes ------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global start_distract_time, alert_triggered, cheating_displayed

    try:
        data = request.json['image']
        encoded = data.split(",")[1]
        img_array = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        gaze_direction = "NO FACE"
        head_direction = "N/A"

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            r_outer = (int(face.landmark[133].x * w), int(face.landmark[133].y * h))
            r_inner = (int(face.landmark[33].x * w), int(face.landmark[33].y * h))
            eye = crop_eye(frame, (r_outer, r_inner))
            if eye.size > 0:
                processed = preprocess_eye(eye)
                pred = model.predict(processed, verbose=0)
                gaze_direction = labels[np.argmax(pred)]

            image_points = np.array([
                (face.landmark[1].x * w, face.landmark[1].y * h),
                (face.landmark[152].x * w, face.landmark[152].y * h),
                (face.landmark[263].x * w, face.landmark[263].y * h),
                (face.landmark[33].x * w, face.landmark[33].y * h),
                (face.landmark[287].x * w, face.landmark[287].y * h),
                (face.landmark[57].x * w, face.landmark[57].y * h)
            ], dtype='double')

            rot_vec, trans_vec = get_head_pose(image_points, frame.shape)
            if rot_vec is not None:
                rmat = cv2.Rodrigues(rot_vec)[0]
                proj = np.hstack((rmat, trans_vec))
                _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
                yaw = euler[1]

                if yaw > 15:
                    head_direction = "LEFT"
                elif yaw < -15:
                    head_direction = "RIGHT"
                else:
                    head_direction = "CENTER"

        now = time.time()

        if is_distracted(gaze_direction, head_direction):
            if start_distract_time is None:
                start_distract_time = now
                alert_triggered = False
                cheating_displayed = False

            elapsed = now - start_distract_time

            if elapsed >= 3 and not alert_triggered:
                play_alert()
                alert_triggered = True

            if elapsed >= 10 and not cheating_displayed:
                cheating_displayed = True
                return jsonify({
                    'gaze': gaze_direction,
                    'head': head_direction,
                    'alert': '⚠️ CHEATING DETECTED!',
                    'focus': 'Please focus here!'
                })

            elif elapsed >= 3:
                return jsonify({
                    'gaze': gaze_direction,
                    'head': head_direction,
                    'focus': 'Please focus here!'
                })

        else:
            start_distract_time = None
            alert_triggered = False
            cheating_displayed = False

        return jsonify({'gaze': gaze_direction, 'head': head_direction})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'gaze': 'ERROR'}), 500

# ------------------ Report Logger ------------------

@app.route('/log_distraction', methods=['POST'])
def log_distraction():
    data = request.json
    candidate_id = data.get("candidate_id", "unknown")
    events = data.get("events", [])
    total_time = data.get("total_time", 0)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{candidate_id}_{timestamp}.json"
    os.makedirs("reports", exist_ok=True)

    with open(os.path.join("reports", filename), 'w') as f:
        json.dump({
            "candidate_id": candidate_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_distraction_time": f"{total_time} seconds",
            "distraction_events": events
        }, f, indent=4)

    return jsonify({"status": "saved", "filename": filename})

# ------------------ Admin Login + Reports ------------------

ADMIN_USERNAME = "vansh"         
ADMIN_PASSWORD = "vansh1234"
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == ADMIN_USERNAME and request.form['password'] == ADMIN_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('reports'))
        return "Invalid credentials", 401
    return render_template('login.html')

@app.route('/reports')
def reports():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    report_files = []
    if os.path.exists("reports"):
        for file in os.listdir("reports"):
            if file.endswith(".json"):
                with open(os.path.join("reports", file)) as f:
                    report_files.append(json.load(f))

    return render_template("reports.html", reports=report_files)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ------------------ Run App ------------------

if __name__ == '__main__':
    app.run(debug=True)
