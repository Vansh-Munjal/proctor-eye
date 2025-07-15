import cv2    #for image processing
import numpy as np

def crop_eye(frame, eye_coords):
    (x1, y1), (x2, y2) = eye_coords  #Takes the full frame and coordinates of the eye corners 
    x1, x2 = min(x1, x2), max(x1, x2) #Safely crops the rectangular eye region between those two points.
    y1, y2 = min(y1, y2), max(y1, y2)
    return frame[y1:y2, x1:x2]

def preprocess_eye(eye_img):
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    return normalized.reshape(1, 64, 64, 1)  # Reshape to match CNN input format

#MAJOR PURPOSE
#Crop the eye from the webcam frame.
#Preprocess the cropped eye image to prepare it for prediction using the CNN model.

#Grayscale is a way of representing an image using shades of gray — with no color