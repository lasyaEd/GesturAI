import cv2 # type: ignore
import mediapipe as mp # type: ignore
import numpy as np


class HandTracker:
    
    def __init__(self, detection_conf=0.5, tracking_conf=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=detection_conf, 
                                         min_tracking_confidence=tracking_conf)
        self.mp_draw = mp.solutions.drawing_utils

    def detect_hands(self, frame):
        """Detects hands and returns processed landmarks."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results

    def draw_landmarks(self, frame, hand_landmarks):
        """Draws landmarks on the detected hands."""
        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    