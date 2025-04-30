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
import src.gesture_model as gm

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


def model_training_pipeline(
    df : pd.DataFrame,
    gesture_index_map : dict,
    sample_size_per_gesture : int = 100,
    sample_size_other_gesture : int = 250):
    
    # Drop rows with NaN values
    df = df.dropna()

    # Convert handedness to numeric values
    df['handedness'] = df['handedness'].map({'Left': -1, 'Right': 1})

    # Convert gesture labels to numeric values
    df['y'] = df['gesture_name'].map(gesture_index_map)
    df = df.drop(columns=['gesture_name'])

    # Balance the sample
    df = gm.get_balanced_sample(
        df,
        sample_size_per_gesture = sample_size_per_gesture,
        sample_size_other_gesture = sample_size_other_gesture
    )

    # Normalize the landmarks
    df = normalize_landmarks_df(df)

    # X,y split
    X = df.drop(columns=["y"])
    y = df["y"]

    # Train and return the model
    return train_model(X,y)


def normalize_landmarks(hand_landmarks):
    """
    Args:
        hand_landmarks: List of landmarks, each landmark is a list of [x, y, z] coordinates.
        
    Returns:
        List of lists of normalized landmarks, where each landmark is a list of [x, y, z] coordinates.
    """

    # Get the average landmark coordinates and center the hand coordinates
    hand_landmarks_centers = []
    for dim in range(3): 
        hand_landmarks_centers.append(np.mean([landmark[dim] for landmark in hand_landmarks]))
    
    for lm_id in range(len(hand_landmarks)):
        for dim in range(3):
            hand_landmarks[lm_id][dim] -= hand_landmarks_centers[dim]
    
    # Get center, min and max for each dimension
    hand_landmarks_centers = []
    hand_landmarks_mins = []
    hand_landmarks_maxs = []
    for dim in range(3):
        hand_landmarks_centers.append(np.mean([landmark[dim] for landmark in hand_landmarks]))
        hand_landmarks_mins.append(np.min([landmark[dim] for landmark in hand_landmarks]))
        hand_landmarks_maxs.append(np.max([landmark[dim] for landmark in hand_landmarks]))
    
    # Then, scale them to fit within the range of -1 to 1
    for lm_id in range(len(hand_landmarks)):
        for dim in range(3):
            # First, center the landmarks around the mean
            hand_landmarks[lm_id][dim] -= hand_landmarks_centers[dim]
            # Then, scale them to fit within the range of -1 to 1
            # based on the distance of the max and min values from the mean
            denom = max(abs(hand_landmarks_maxs[dim]-hand_landmarks_centers[dim]), 
                        abs(hand_landmarks_mins[dim]-hand_landmarks_centers[dim]))
            if denom > 0:
                hand_landmarks[lm_id][dim] /= denom
            else:
                hand_landmarks[lm_id][dim] = 0

    return hand_landmarks


def get_balanced_sample(
    df : pd.DataFrame,
    sample_size_per_gesture : int = 100,
    sample_size_other_gesture : int = 250):

    # Create an empty dataframe to store selected samples
    df_balanced = pd.DataFrame(columns=df.columns)

    # For each handedness and gesture, select observations and add to df_sampled
    for handedness in [-1, 1]:
        for gesture in df['y'].unique():
            sample_size = sample_size_other_gesture if gesture == 'other_gesture' else sample_size_per_gesture
            df_temp = df[(df['handedness'] == handedness) & (df['y'] == gesture)]

            # Check if we have enough samples
            if len(df_temp) >= sample_size:
                subsample = df_temp.sample(n=sample_size, random_state=42)
                df_balanced = pd.concat([df_balanced, subsample], ignore_index=True)
            else:
                print(f"Not enough samples for gesture {gesture} with handedness = {handedness}.")
                return None
            
    df_balanced = df_balanced.reset_index(drop=True)


    return df_balanced

# Wrapper function to normalize each set of landmarks in a dataframe
# using the normalize_landmarks function
def normalize_landmarks_df(df):

    df = df.copy()

    # For each row of the dataframe, normalize the landmarks
    for index, row in df.iterrows():
        # Extract landmarks from the row
        hand_landmarks = []
        for i in range(21):
            hand_landmarks.append([row[f'{i}_x'], row[f'{i}_y'], row[f'{i}_z']])
        
        # Normalize the landmarks
        normalized_landmarks = normalize_landmarks(hand_landmarks)
        
        # Update the dataframe with normalized landmarks
        for i in range(21):
            df.at[index, f'{i}_x'] = normalized_landmarks[i][0]
            df.at[index, f'{i}_y'] = normalized_landmarks[i][1]
            df.at[index, f'{i}_z'] = normalized_landmarks[i][2]
    
    return df


# Takes a dataframe with columns ['x', 'y', 'z'] for each landmark
# A numeric column 'handedness' indicating the handedness of the gesture (1 for right, -1 for left) 
# and a numeric label column 'y'
# Returns a newly trained model
def train_model(X: pd.DataFrame, y: pd.DataFrame):

    # Split data (stratified ensures class balance in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )  

    # Initialize model
    input_size = X.shape[1]
    num_classes = len(set(y))
    model = GestureClassifier(input_size, num_classes)

    # Training config
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    epochs = 30 + num_classes * 3

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values.astype(np.float32), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values.astype(np.int64), dtype=torch.int64)
    X_test_tensor = torch.tensor(X_test.values.astype(np.float32), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values.astype(np.int64), dtype=torch.int64)

    # Train model
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {loss.item():.4f}")
        

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            predicted = torch.argmax(test_outputs, dim=1)
            accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
            print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
    
            # # Confusion matrix
            # cm = confusion_matrix(y_test, predicted)
            # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            # disp.plot(cmap="Blues")
            # plt.title("Confusion Matrix")
            # plt.xlabel("Predicted")
            # plt.ylabel("True")
            # plt.show()
        

    return model








