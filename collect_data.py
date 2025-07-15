import cv2
import mediapipe as mp
import os

label = input("Enter label (LEFT / CENTER / RIGHT): ").upper()
folder = f"data/{label}"
os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
count = 0
MAX_IMAGES = 500
start_collecting = False

print("🔄 Press 's' once to start capturing images. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        start_collecting = True
        print("✅ Started collecting images...")

    if key == ord('q'):
        print("⛔ Stopped by user.")
        break

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            lm = face.landmark
            r1 = (int(lm[33].x * w), int(lm[33].y * h))
            r2 = (int(lm[133].x * w), int(lm[133].y * h))

            x1, x2 = min(r1[0], r2[0]), max(r1[0], r2[0])
            y1, y2 = min(r1[1], r2[1]), max(r1[1], r2[1])

            eye = frame[y1:y2, x1:x2]
            if eye.size > 0 and start_collecting and count < MAX_IMAGES:
                eye = cv2.resize(eye, (64, 64))
                filename = f"{folder}/{label}_{count}.jpg"
                cv2.imwrite(filename, eye)
                count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.putText(frame, f"{label} Count: {count}/{MAX_IMAGES}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(frame, "'s' to Start | 'q' to Quit", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Collecting Eye Data", frame)

    if count >= MAX_IMAGES:
        print(f"✅ Done: Collected {MAX_IMAGES} images for {label}.")
        break

cap.release()
cv2.destroyAllWindows()
