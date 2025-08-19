#OPEN @TRAIN_KNN.PY FOR INTRUCTIONS ON HOW TO RUN 


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs

import cv2
import numpy as np
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier
import pickle

# ------------------- CONSTANTS -------------------
MODEL_PATH = "expression_knn_model.pkl"  # Path to trained model
EMOTIONS = ["Angry", "Happy", "Neutral", "Sad"]  # Must match training order
IMG_SIZE = 24     # Image size used for training
WAVE_THRESHOLD = 500000  # Threshold for motion detection (wave gesture)

# ------------------- LOAD FACE DETECTOR -------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ------------------- LOAD MEDIAPIPE HAND DETECTION -------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ------------------- LOAD TRAINED MODEL -------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"[ERROR] Model file '{MODEL_PATH}' not found. "
        "Please run train_knn.py first to train the model."
    )

with open(MODEL_PATH, "rb") as f:
    knn = pickle.load(f)

# ------------------- HAND GESTURE FUNCTIONS -------------------
def get_finger_state(hand_landmarks):
    """Returns a list of finger states (1 = extended, 0 = folded)."""
    fingers = []

    # Thumb check (horizontal position)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other four fingers check (vertical position)
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def detect_gesture(hand_landmarks):
    """Detects simple hand gestures based on finger states."""
    fingers = get_finger_state(hand_landmarks)

    if fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    elif fingers == [0, 1, 1, 0, 0]:
        return "Peace Sign"
    elif fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    else:
        return "Hand Detected"

# ------------------- START WEBCAM -------------------
cap = cv2.VideoCapture(0)
prev_gray = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Motion detection for wave
    motion = cv2.absdiff(gray, prev_gray)
    motion = cv2.GaussianBlur(motion, (5, 5), 0)
    _, motion_mask = cv2.threshold(motion, 25, 255, cv2.THRESH_BINARY)
    motion_detected = np.sum(motion_mask[0:200, 0:200]) > WAVE_THRESHOLD
    prev_gray = gray.copy()

    # Face + Emotion detection
    for (x, y, w, h) in faces:
        roi = cv2.resize(gray[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE)).flatten()
        prediction = knn.predict([roi])[0]
        emotion = EMOTIONS[prediction]

        # Draw face box & emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 200, 0), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if motion_detected:
            cv2.putText(frame, "Wave Detected", (x, y+h+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Hand gesture detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks)
            cv2.putText(frame, gesture, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show video feed
    cv2.imshow("Emotion + Gesture Tracker", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
