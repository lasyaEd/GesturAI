import torch.nn as nn
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import mediapipe as mp

def load_gesture_data(path, gesture_labels):
     # Load the data for each gesture
        # CSV format,
        # Each file is all of the data for a specific gesture
        # Each row represents a frame of landmarks
    
    for gesture in gesture_labels:
        gesture_path = f"{path}/{gesture}"
       

        # TODO load the data 


# # Standardize preprocessing of the landmarks to ensure consistent input to the model
# def normalize_landmarks(landmarks):
#         keypoints = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
#         keypoints -= np.mean(keypoints, axis=0)
#         keypoints /= np.std(keypoints, axis=0) # TODO this might not be the best way to normalize
#         return keypoints.flatten()


def normalize_landmarks(hand_landmarks):
    """
    Normalizes hand landmarks from MediaPipe's NormalizedLandmarkList object.
    
    Args:
        hand_landmarks: List of landmarks, each landmark is a list of [x, y, z] coordinates.
        
    Returns:
        Flattened numpy array of normalized landmarks
    """

    # Get the average landmark coordinates and center the hand coordinates
    hand_landmarks_centers = []
    for dim in range(3): 
        hand_landmarks_centers.append(np.mean([landmark[dim] for landmark in hand_landmarks]))
    
    for lm_id in range(len(hand_landmarks)):
        for dim in range(3):
            hand_landmarks[lm_id][dim] -= hand_landmarks_centers[dim]
    
    # Get min and max for each dimension
    hand_landmarks_mins = []
    hand_landmarks_maxs = []
    for dim in range(3):
        hand_landmarks_mins.append(np.min([landmark[dim] for landmark in hand_landmarks]))
        hand_landmarks_maxs.append(np.max([landmark[dim] for landmark in hand_landmarks]))
    
    # Normalize to range [0, 1]
    # TODO Switch to normalizing to range [-1, 1] as soon as model is trained this way
    for lm_id in range(len(hand_landmarks)):
        for dim in range(3):
            # Handle case where max and min are the same (division by zero)
            range_val = hand_landmarks_maxs[dim] - hand_landmarks_mins[dim]
            if range_val > 0:
                hand_landmarks[lm_id][dim] = (hand_landmarks[lm_id][dim] - hand_landmarks_mins[dim]) / range_val
            else:
                hand_landmarks[lm_id][dim] = 0  # Default to 0 if all values are identical
    
    return np.array(hand_landmarks).flatten()


# TODO add a function to add handedness to the keypoints, transform to tensor, etc
# TODO add dropout layers to the model to prevent overfitting
# TODO add batch normalization to the model to improve training speed and stability
# TODO turn this into a simple sequential model to simplify the code?
# TODO switch to leaky relu?
class GestureClassifier(nn.Module):
    def __init__(self, input_features, output_classes):
        super(GestureClassifier, self).__init__()
        self.fc1 = nn.Linear(input_features, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, output_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# model training function
def train_gesture_model(
    data : pd.DataFrame,
    gesture_index_map : dict,
    sample_size_per_gesture : int = 100,
    sample_size_other_gesture : int = 250,):

    # Drop all rows with any nulls - shouldn't be any
    df = data.dropna()

    # Create an empty dataframe to store selected samples
    df_sampled = pd.DataFrame(columns=df.columns)

    # For each handedness and gesture, select observations and add to df_sampled
    for right_hand in ['-1', '1']:
        for gesture in df['gesture_name'].unique():
            sample_size = sample_size_other_gesture if gesture == 'other_gesture' else sample_size_per_gesture
            df_temp = df[(df['right_hand'] == right_hand) & (df['gesture_name'] == gesture)]

            # Check if we have enough samples
            if len(df_temp) >= sample_size:
                sampled = df_temp.sample(n=sample_size, random_state=42)
                df_sampled = pd.concat([df_sampled, sampled])
            else:
                # Take all available if less than required
                df_sampled = pd.concat([df_sampled, df_temp])

    # replace and fix up df
    df = df_sampled
    df = df.reset_index(drop=True)

    # Create new feature y based on gesture index map
    # TODO test
    df['y'] = df['gesture_name'].map(gesture_index_map)
    df = df.drop(columns=["gesture_name", "capture_number"])

    # Get mean of each coordinate
    df['center_x'] = np.mean([df[f'{i}_x'] for i in range(21)], axis=0)
    df['center_y'] = np.mean([df[f'{i}_y'] for i in range(21)], axis=0)
    df['center_z'] = np.mean([df[f'{i}_z'] for i in range(21)], axis=0)


    #Translate coordinates to base on center
    for i in range(21):
        df[f'rec_{i}_x'] = df[f'{i}_x'] - df[f'center_x']
        df[f'rec_{i}_y'] = df[f'{i}_y'] - df[f'center_y']
        df[f'rec_{i}_z'] = df[f'{i}_z'] - df[f'center_z']

    # Scale the landmarks to fit within a range of 0 to 1
    for dim in ['x','y','z']:
        df[f'rec_max_{dim}'] = np.max([df[f'rec_{i}_{dim}'] for i in range(21)])
        df[f'rec_min_{dim}'] = np.min([df[f'rec_{i}_{dim}'] for i in range(21)])
        for i in range(21):
            df[f'scaled_{i}_{dim}'] = (df[f'rec_{i}_{dim}'] - df[f'rec_min_{dim}']) / (df[f'rec_max_{dim}'] - df[f'rec_min_{dim}'])
            df = df.copy() # to avoid excessive fragmentation

    # Establish feature set
    feature_set = [f'scaled_{i}_{j}' for i in range(21) for j in ['x','y','z']]
    feature_set.append('handedness_encoded')

    X = df.drop["y"]
    y = df["y"]

    # Split data (stratified ensures class balance in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
        
    # Initialize model
    input_size = X.shape[1]
    num_classes = len(set(y))
    model = GestureClassifier(input_size, num_classes)

    # Training config
    # TODO add a scheduler, early stopping, optimal model identification, etc
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    epochs = 30

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Train model
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        predicted = torch.argmax(test_outputs, dim=1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

        # Confusion matrix
        cm = confusion_matrix(y_test, predicted)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    # TODO model selection - review model results and return the best modle
    # TODO consider failure conditions - when should the new gesture data be rejected as inadequte to train a new model?
    return model
    


