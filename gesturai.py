import cv2
import mediapipe as mp
import numpy as np
import torch
import pickle
import time
import os
import json
import sys
import src.gesture_model as gm
import src.actions as actions
import src.crud as crud
from dotenv import load_dotenv


# Load environment variables
# These should be set in the environment or a .env file
load_dotenv()
GESTURE_INDEX_MAP_PATH = os.environ["GESTURAI_GESTURE_INDEX_MAP_PATH"]
GESTURE_ACTION_MAP_PATH = os.environ["GESTURAI_GESTURE_ACTION_MAP_PATH"]
GESTURE_MODEL_PATH = os.environ["GESTURAI_GESTURE_MODEL_PATH"]

# Load label map
gesture_index_map = crud.load_json_mapping(GESTURE_INDEX_MAP_PATH)[0]
index_gesture_map = {v: k for k, v in gesture_index_map.items()}

# Load gesture recognition model
# gesture_model = GestureClassifier(input_size=64, num_classes=len(gesture_index_map.keys()))
# gesture_model.load_state_dict(torch.load(GESTURE_MODEL_PATH))
# gesture_model.eval()

# TODO change this to load model weights only when the model is trained with weights_only = True
#load gesture model including structure from pth file
gesture_model = torch.load(GESTURE_MODEL_PATH, map_location=torch.device('cpu'), weights_only = False)

# Load gesture-action mapping
gesture_action_map = crud.load_json_mapping(GESTURE_ACTION_MAP_PATH)[0]

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
mp_draw = mp.solutions.drawing_utils

# Prep main loop
active_context = "mode_context_selection"
cap = cv2.VideoCapture(0)
cooldown = 2.5  # seconds
last_trigger_time = time.time()
gesture_display = "Waiting..."

# main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for i in range(len(result.multi_hand_landmarks)):
            
            # Get the hand landmarks and handedness
            hand_landmarks = result.multi_hand_landmarks[i]
            # TODO Change this to -1 left, 1 right as soon as a new model has been trained this way
            handedness = 1 if result.multi_handedness[i].classification[0].label == "Right" else 0
            
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks and convert to 3D coordinates for normalization
            hand_landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]

            # Normalize the landmarks, returning a 1D numpy array          
            hand_landmarks = gm.normalize_landmarks(hand_landmarks)
            # appen handedness as a numeric feature for the model
            hand_landmarks = np.append(hand_landmarks, handedness)

            # Convert to tensor & predict gesture
            hand_landmarks_tensor = torch.tensor(hand_landmarks, dtype=torch.float32)
            with torch.no_grad():
                gesture_output = gesture_model(hand_landmarks_tensor)
                predicted_gesture = index_gesture_map[torch.argmax(gesture_output).item()]
            print(f"Predicted gesture: {predicted_gesture}")

            try:
                                
                # Set display text to current gesture and context
                gesture_display = f"{predicted_gesture} ({active_context})"

                # Check if the gesture maps to an action in the active context
                action_name = "None"
                try:
                    action_name = gesture_action_map[active_context][predicted_gesture]
                except KeyError:
                    print(f"⚠️ No action found for gesture '{predicted_gesture}' in '{active_context}'")

                if time.time() - last_trigger_time > cooldown and action_name != "None":
                    
                    # If the first 5 letters of the action name are "mode_"
                    # And the action name is one of the keys in the gesture_map (a valid context)
                    # Then switch the active context to the action name
                    if action_name[:5] == "mode_" and action_name in gesture_action_map.keys():
                        print(f"🔄 Context switched to: {active_context}")
                        active_context = gesture_action_map[active_context][predicted_gesture]
                        last_trigger_time = time.time()
                            
                    # Map gesture to action, and execute if applicable
                    else:
                        
                        try:
                            print(f"[ACTION] {predicted_gesture} in {active_context}")
                            action = getattr(actions, action_name)
                            action()
                            last_trigger_time = time.time()
                        except AttributeError:
                            print(f"⚠️ Action function '{action_name}' not found in actions.py")        

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
