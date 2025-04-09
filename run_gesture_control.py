import cv2
import mediapipe as mp
import numpy as np
import torch
import pickle
import time
from model import GestureClassifier
from gesture_mapping import load_gesture_mapping, get_action_from_gesture

# Load model
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)  # e.g., {0: 'open_palm', 1: 'fist', ...}

model = GestureClassifier(input_size=63, num_classes=len(label_map))
model.load_state_dict(torch.load("gesture_model.pth"))
model.eval()

# Context switching triggers
context_triggers = {
    "three_fingers": "word_mode",
    "peace_sign": "media_mode",
    "ok_sign": "presentation_mode",
    "rock_on": "default"
}
active_context = "default"

# Load gesture-action mapping
gesture_map = load_gesture_mapping()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
mp_draw = mp.solutions.drawing_utils

def normalize_keypoints(landmarks):
    keypoints = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
    keypoints -= np.mean(keypoints, axis=0)
    keypoints /= np.std(keypoints, axis=0)
    return keypoints.flatten()

# Webcam loop
cap = cv2.VideoCapture(0)
cooldown = 3
last_trigger_time = 0
gesture_display = "Waiting..."

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            try:
                keypoints = normalize_keypoints(hand_landmarks.landmark)
                input_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)
                output = model(input_tensor)
                predicted_idx = torch.argmax(output, dim=1).item()
                gesture = label_map.get(predicted_idx, "unknown")

                current_time = time.time()
                gesture_display = f"{gesture} ({active_context})"

                # Handle mode switching gestures
                if current_time - last_trigger_time > cooldown:
                    if gesture in context_triggers:
                        new_context = context_triggers[gesture]
                        if new_context != active_context:
                            active_context = new_context
                            print(f"🔄 Context switched to: {active_context}")
                            last_trigger_time = current_time
                        continue

                    # Use user-defined mapping
                    action = get_action_from_gesture(gesture, active_context, gesture_map)
                    if action:
                        print(f"[ACTION] {gesture} in {active_context}")
                        action()
                        last_trigger_time = current_time

            except Exception as e:
                print(f"[ERROR] {e}")
    else:
        gesture_display = f"No hand detected ({active_context})"

    # Display gesture and context
    cv2.putText(frame, f"Gesture: {gesture_display}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Mode: {active_context}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("GesturAI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
