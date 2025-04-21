import cv2
import mediapipe as mp
import pandas as pd


def capture_gesture_data(gesture_name):

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Configure MediaPipe Hands with default settings
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    num_landmarks = 21

    # Initialize the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution to HD (1280x720) for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Press 'c' to capture current landmarks")
    print ("Press 'r' reset captured landmarks")
    print("Press 's' to save the captured landmarks")
   
    # Data collection application
    capture_number = 0

    # Create landmarks header
    landmarks_header = [
        'gesture_name',
        'capture_number', 
        'right_hand',
        'score']
    for landmark_id in range(num_landmarks):
        for point_id in ['x', 'y', 'z']:
            landmarks_header.append(f'{landmark_id}_{point_id}')
    
    landmarks_df = pd.DataFrame(columns=landmarks_header)

    # Main loop to process video frames and capture landmarks
    while cap.isOpened():
        # Read frame from webcam
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam")
            break
        
        # Flip the frame horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR image to RGB (CV2 uses BGR but MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        landmarks = hands.process(rgb_frame)
        
        # Create a copy of the frame to draw on
        display_frame = frame.copy()
        

        # Draw hand landmarks if detected
        if landmarks.multi_hand_landmarks:
            for hand_landmarks in landmarks.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        cv2.putText(
            display_frame, 
            f'Capturing {gesture_name} #{capture_number}',
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2)

        cv2.imshow("MediaPipe Hands Capture", display_frame)

        
        # Handle keyboard input 
        # masking to capture only the lower 8 bits to help with cross-platform compatibility
        key = cv2.waitKey(1) & 0xFF
        
        # 'c' key to capture current frame and landmarks
        # TODO add a check to see if all landmarks are present - don't capture bad data
        if key == ord('c'):
            if landmarks.multi_hand_world_landmarks:

                landmarks_data = {
                    'gesture_name': gesture_name,
                    'score': landmarks.multi_handedness[0].classification[0].score,
                    'capture_number': capture_number, 
                    'right_hand': 1 if landmarks.multi_handedness[0].classification[0].label == 'Right' else -1,
                }          
                for landmark_id in range(num_landmarks):
                    landmark = landmarks.multi_hand_landmarks[0].landmark[landmark_id]
                    landmarks_data[f'{landmark_id}_x'] = landmark.x
                    landmarks_data[f'{landmark_id}_y'] = landmark.y
                    landmarks_data[f'{landmark_id}_z'] = landmark.z
                        
                # Append landmarks data to dataframe
                landmarks_df = pd.concat([landmarks_df, pd.DataFrame([landmarks_data])], ignore_index=True)
                
                capture_number += 1
                print("Frame captured with hand landmarks")
            else:
                print("No hands detected. Position your hand in the frame.")

        elif key == ord('r'):
            # Reset captured landmarks
            landmarks_df = pd.DataFrame(columns=landmarks_header)
            capture_number = 0
            print("Reset captured landmarks")

        # 's' key to stop and return captured landmarks
        elif key == ord('s'):  
            
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            
            # Return the captured landmarks DataFrame
            return landmarks_df

 
    
    