import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
with open("gesture_data.pkl", "rb") as f:
    dataset = pickle.load(f)

X = np.array([item[0] for item in dataset])  # Normalized keypoints
y = np.array([item[1] for item in dataset])  # Gesture labels

# Split data (stratified ensures class balance in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define model architecture
class GestureClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GestureClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
input_size = X.shape[1]
num_classes = len(set(y))
model = GestureClassifier(input_size, num_classes)

# Training config
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 65

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

# Save model
torch.save(model.state_dict(), "gesture_model.pth")
print("🎉 Model saved to gesture_model.pth")

# Save label map
GESTURE_LABELS = {
    'open_palm': 0,
    'fist': 1,
    'peace_sign': 2,
    'thumbs_up': 3,
    'pointing_finger': 4,
    'ok_sign': 5,
    'thumbs_down': 6,
    'three_fingers': 7,
    'rock_on': 8,
    'pinch': 9,
    'swipe_left': 10,
    'swipe_right': 11
}
label_map = {v: k for k, v in GESTURE_LABELS.items()}
with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)
print("📌 Label map saved to label_map.pkl")
