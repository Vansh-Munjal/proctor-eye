import cv2
import numpy as np

# Predefined 3D model points of the face
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -330.0, -65.0),   # Chin
    (-225.0, 170.0, -135.0),# Left eye left corner
    (225.0, 170.0, -135.0), # Right eye right corner
    (-150.0, -150.0, -125.0),# Left mouth corner
    (150.0, -150.0, -125.0) # Right mouth corner
], dtype=np.float32)

def get_head_pose(image_points, frame_size):
    focal_length = frame_size[1]
    center = (frame_size[1] / 2, frame_size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs
    )

    return rotation_vector, translation_vector


# solvePnP() solves the Perspective-n-Point problem: it estimates:
# rotation_vector: how the head is rotated (pitch, yaw, roll)
# translation_vector: where the head is located in space