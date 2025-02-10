import cv2 # type: ignore
from hand_tracker import HandTracker
from gesture_utils import is_fist
from actions import open_terminal

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


            # Check if a fist is detected
            if is_fist(hand_landmarks.landmark):
                print("Fist Detected - Opening Terminal!")
                open_terminal()

    cv2.imshow("Hand Gesture Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
