import numpy as np

def is_fist(hand_landmarks):
    """Detects if the hand is making a fist based on fingertip distances."""
    if hand_landmarks is None:
        return False

    tip_ids = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
    palm_id = 0  # Wrist

    distances = []
    for tip in tip_ids:
        tip_x, tip_y = hand_landmarks[tip].x, hand_landmarks[tip].y
        palm_x, palm_y = hand_landmarks[palm_id].x, hand_landmarks[palm_id].y
        distance = np.linalg.norm(np.array([tip_x, tip_y]) - np.array([palm_x, palm_y]))
        distances.append(distance)

    avg_distance = np.mean(distances)  # Compute the average fingertip distance
    print(f"Fist detection avg distance: {avg_distance}")  # Debugging output

    return avg_distance < 0.20  # Adjusted threshold for better recognition

def is_open_palm(landmarks):
    """Detect an open palm (all fingers extended)."""
    return all(landmarks[i].y < landmarks[i-2].y for i in [8, 12, 16, 20])  # Finger tips above knuckles

def is_thumb_up(landmarks):
    """Detect a thumbs-up gesture."""
    return landmarks[4].y < landmarks[3].y and all(landmarks[i].y > landmarks[6].y for i in [8, 12, 16, 20])

def is_thumb_down(landmarks):
    """Detect a thumbs-down gesture."""
    return landmarks[4].y > landmarks[3].y and all(landmarks[i].y < landmarks[6].y for i in [8, 12, 16, 20])

def is_two_fingers(landmarks):
    """Detects index and middle fingers up, others down."""
    return (
        landmarks[8].y < landmarks[6].y and   # index up
        landmarks[12].y < landmarks[10].y and  # middle up
        landmarks[16].y > landmarks[14].y and  # ring down
        landmarks[20].y > landmarks[18].y      # pinky down
    )

def is_ok_sign(landmarks):
    """Detects if thumb and index fingertips are touching (OK sign)."""
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    index_tip = np.array([landmarks[8].x, landmarks[8].y])
    distance = np.linalg.norm(thumb_tip - index_tip)
    return distance < 0.05  # tune threshold based on real input


def is_pinch(landmarks):
    """Detects if thumb tip is pinching any of the other fingertips."""
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    finger_tips = [8, 12, 16, 20]  # index, middle, ring, pinky

    for tip in finger_tips:
        fingertip = np.array([landmarks[tip].x, landmarks[tip].y])
        distance = np.linalg.norm(thumb_tip - fingertip)
        if distance < 0.04:  # tune threshold
            return True
    return False

def is_peace(landmarks):
    """Detects a peace sign gesture (index and middle up, others down)."""
    return (
        landmarks[8].y < landmarks[6].y and    # index up
        landmarks[12].y < landmarks[10].y and  # middle up
        landmarks[16].y > landmarks[14].y and  # ring down
        landmarks[20].y > landmarks[18].y      # pinky down
    )


# Global or class-level variable to store previous x
prev_x = None

def is_swipe_left(landmarks):
    global prev_x
    current_x = landmarks[0].x  # wrist x

    if prev_x is not None and prev_x - current_x > 0.1:
        prev_x = current_x
        return True
    prev_x = current_x
    return False

def is_swipe_right(landmarks):
    global prev_x
    current_x = landmarks[0].x

    if prev_x is not None and current_x - prev_x > 0.1:
        prev_x = current_x
        return True
    prev_x = current_x
    return False

def detect_gesture(landmarks):
    if landmarks is None:
        return "none"

    if is_fist(landmarks):
        return "fist"
    if is_thumb_up(landmarks):
        return "thumbs_up"
    if is_thumb_down(landmarks):
        return "thumbs_down"
    if is_peace(landmarks):
        return "peace"
    if is_two_fingers(landmarks):
        return "two_fingers"
    if is_open_palm(landmarks):
        return "palm"
    if is_ok_sign(landmarks):
        return "ok_sign"
    if is_pinch(landmarks):
        return "pinch"
    if is_swipe_left(landmarks):
        return "swipe_left"
    if is_swipe_right(landmarks):
        return "swipe_right"

    return "unknown"
