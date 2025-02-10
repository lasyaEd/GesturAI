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

    return avg_distance < 0.30  # Adjusted threshold for better recognition
