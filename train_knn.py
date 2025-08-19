#THIS PROJECT IS FOR DECTION OF CERTAIN HAND GESTURES AND FACE RECOGNITION, ALONG WITH EMOTIONS.
#IF YOU WANT MUCH BETTER EMOTION RECOGNITION TURN THE #IMG_SIZE TO 96 OR HIGHER FOR BETTER ACCURACY
#THIS FILE REQUIRE PYTHON 3.10 SO USE THAT VERSION OR VENV
#ALSO CHANGE IMG_SIZE IN MAIN.PY , DELETE THE EXPRESSION_KNN_MODEL AND THEN RUN TRAIN_KNN.PY FOLLOWED BY MAIN.PY
import os
import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

# Path to the dataset folders (must be in the same directory as this script)
TRAIN_DIR = "train"  # Folder with subfolders for each emotion
MODEL_PATH = "expression_knn_model.pkl"  # File where model will be saved
IMG_SIZE = 24  # Image size (96x96) for training — should match main.py

# Emotions you want to classify — these should match the folder names in 'train'
EMOTIONS = ["Angry", "Happy", "Neutral", "Sad"]

# Map emotion name → numeric label
label_map = {name: idx for idx, name in enumerate(EMOTIONS)}

# Arrays to store training data
X = []
y = []

print("[INFO] Loading training images...")

# Loop through each emotion folder
for emotion_name in EMOTIONS:
    folder_path = os.path.join(TRAIN_DIR, emotion_name)
    
    if not os.path.exists(folder_path):
        print(f"[WARNING] Folder {folder_path} does not exist, skipping...")
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        # Read the image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue  # Skip broken or unreadable files

        # Resize image to fixed size
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Flatten image (convert 2D pixels into a 1D array)
        X.append(img.flatten())

        # Append the corresponding label
        y.append(label_map[emotion_name])

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

print(f"[INFO] Loaded {len(X)} images for training.")

# Create and train the KNN model
print("[INFO] Training KNN model...")
knn = KNeighborsClassifier(n_neighbors=3)  # You can change k for tuning
knn.fit(X, y)

# Save trained model to disk
with open(MODEL_PATH, "wb") as f:
    pickle.dump(knn, f)

print(f"[INFO] Model saved to {MODEL_PATH}")
