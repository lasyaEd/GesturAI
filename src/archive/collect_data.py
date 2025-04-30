import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import defaultdict

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
mp_draw = mp.solutions.drawing_utils

# Define gesture labels mapped to their intended system commands
GESTURE_LABELS = {
    'open_palm': 0,         # Paste (Cmd+V / Ctrl+V)
    'fist': 1,              # Open Terminal
    'peace_sign': 2,        # Copy (Cmd+C / Ctrl+C)
    'thumbs_up': 3,         # Open Browser / Undo
    'pointing_finger': 4,   # Select All (Cmd+A / Ctrl+A)
    'ok_sign': 5,           # Lock Computer / Presentation Mode
    'thumbs_down': 6,       # File Explorer / Redo
    'three_fingers': 7,     # Word Mode activation
    'rock_on': 8,           # Default Mode
    'pinch': 9,             # Pause/Play
    'swipe_left': 10,       # Decrease Volume / Tab Left
    'swipe_right': 11       # Increase Volume / Tab Right
}

DATASET_PATH = "/Users/lasyaedunuri/Documents/Computer Vision/GesturAI/gesture_data.pkl"

def normalize_keypoints(hand_landmarks):
    """Centers and scales keypoints to make them model-friendly."""
    keypoints = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
    keypoints -= np.mean(keypoints, axis=0)
    keypoints /= np.std(keypoints, axis=0)
    return keypoints.flatten()

def load_existing_data():
    try:
        with open(DATASET_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []

def count_existing_samples(data):
    label_counts = defaultdict(int)
    for _, label in data:
        label_counts[label] += 1
    return label_counts

def collect_data():
    cap = cv2.VideoCapture(0)
    collected_data = load_existing_data()
    label_counts = count_existing_samples(collected_data)

    for gesture, label in GESTURE_LABELS.items():
        existing_count = label_counts.get(label, 0)
        if existing_count >= 200:
            print(f"✅ Skipping '{gesture}' (already has {existing_count} samples)")
            continue

        print(f"\n🎯 Collecting data for '{gesture}' (Label ID: {label})")
        print(f"Existing samples: {existing_count}/200")
        input("👉 Position your hand and press Enter to start...")

        frame_count = 0
        while frame_count < (200 - existing_count):
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame from webcam.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    keypoints = normalize_keypoints(hand_landmarks)
                    collected_data.append((keypoints, label))
                    frame_count += 1

                    print(f"✅ Captured frame {existing_count + frame_count}/200 for '{gesture}'")
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"Gesture: {gesture} ({existing_count + frame_count}/200)", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Data Collection", frame)

            time.sleep(0.1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

    with open(DATASET_PATH, "wb") as f:
        pickle.dump(collected_data, f)

    print("\n✅ Data collection complete! Saved to gesture_data.pkl.")

if __name__ == "__main__":
    collect_data()
