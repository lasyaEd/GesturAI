import cv2
from hand_tracker import HandTracker
from gesture_utils import is_fist, is_open_palm, is_thumb_up, is_thumb_down
from actions import open_terminal, open_browser, increase_volume, decrease_volume, take_screenshot

# Initialize HandTracker
tracker = HandTracker()

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = tracker.detect_hands(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            tracker.draw_landmarks(frame, hand_landmarks)
            print("Hand detected!")  # Debugging output

            # Check for different gestures and execute corresponding action
            if is_fist(hand_landmarks.landmark):
                print("Fist Detected - Opening Terminal!")
                open_terminal()
            elif is_open_palm(hand_landmarks.landmark):
                print("Open Palm Detected - Opening Browser!")
                open_browser()
            elif is_thumb_up(hand_landmarks.landmark):
                print("Thumbs Up Detected - Increasing Volume!")
                increase_volume()
            elif is_thumb_down(hand_landmarks.landmark):
                print("Thumbs Down Detected - Decreasing Volume!")
                decrease_volume()

    cv2.imshow("Hand Gesture Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
