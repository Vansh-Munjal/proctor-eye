import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/gaze_cnn.h5")
labels = ["LEFT", "CENTER", "RIGHT"]

def predict_gaze(frame):
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    input_data = resized.reshape(1, 64, 64, 1) / 255.0
    prediction = model.predict(input_data, verbose=0)
    return labels[np.argmax(prediction)]
