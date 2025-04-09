import cv2
import time
from hand_tracker import HandTracker
from gesture_utils import detect_gesture  # central detection logic
from actions import gesture_action_map    # maps gestures to functions

# Initialize HandTracker
tracker = HandTracker()

# Start video capture
cap = cv2.VideoCapture(0)

# Cooldown config
cooldown_time = 2  # seconds between triggering gestures
last_action_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = tracker.detect_hands(frame)

    if results.multi_hand_landmarks:
        current_time = time.time()

        for hand_landmarks in results.multi_hand_landmarks:
            tracker.draw_landmarks(frame, hand_landmarks)

            if current_time - last_action_time > cooldown_time:
                # Detect gesture from landmarks
                gesture = detect_gesture(hand_landmarks.landmark)
                print(f"[INFO] Detected Gesture: {gesture}")

                # Check if gesture has an action mapped
                action = gesture_action_map.get(gesture)
                if action:
                    print(f"[ACTION] Executing action for '{gesture}'")
                    action()
                    last_action_time = current_time
                else:
                    print(f"[INFO] No action mapped for gesture: {gesture}")

    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
