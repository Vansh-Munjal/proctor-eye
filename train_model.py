import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Label map
label_map = {"LEFT": 0, "CENTER": 1, "RIGHT": 2}
data = []
labels = []  #labels will store their corresponding directions (0, 1, 2)


# Load images from data folders
# Reads images from data/LEFT, data/CENTER, data/RIGHT folders.
# Converts them to grayscale, resizes to 64×64, and normalizes to [0,1].
# Appends image data and label to their respective lists.
for label in label_map:
    path = f"data/{label}"
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64)) / 255.0
        data.append(img)
        labels.append(label_map[label])

# Convert to arrays
X = np.array(data).reshape(-1, 64, 64, 1)
y = to_categorical(np.array(labels), num_classes=3)   # One-hot encode labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes
])

# Compile & train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

# Save model
model.save("models/gaze_cnn.h5")
print("✅ Model saved as models/gaze_cnn.h5")
