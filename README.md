# 🖐️ Hand Gesture-Based System Control

🚀 **A real-time hand gesture recognition system** that detects gestures using **Mediapipe** and executes system commands like opening a terminal. This modular project makes it easy to extend for other gestures and actions.

---

## 📂 Project Structure
```
hand_gesture_control/
│── main.py           # Runs the main loop
│── hand_tracker.py   # Handles hand tracking using Mediapipe
│── gesture_utils.py  # Defines gesture recognition functions
│── actions.py        # Maps gestures to system commands
│── README.md         # Project documentation
```

---

## 📌 Features
✅ **Real-time Hand Tracking** using **Mediapipe**  
✅ **Fist Detection Gesture** → Opens a terminal  
✅ **Cross-Platform Support** (macOS, Linux, Windows)  
✅ **Modular Codebase** for adding more gestures  

---

## 🛠️ Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/lasyaEd/GesturAI.git
cd GestureAI
```

### 2️⃣ Install Dependencies
Ensure Python 3.11 is installed, then install required libraries:

```bash
pip install opencv-python mediapipe numpy
```

---

## 🚀 How to Run the Project
```bash
python main.py
```
📌 **Usage:**  
- Make a **fist** ✊ → Terminal should open  
- Press **'q'** to exit the program  

---

## 📌 Troubleshooting
❌ **Issue: No module named 'cv2' or 'mediapipe'**  
✅ Solution: Run  
```bash
pip install opencv-python mediapipe numpy
```

❌ **Issue: Terminal doesn't open on macOS/Linux**  
✅ Solution: Uncomment the correct terminal command in **`actions.py`**  

---

## 📜 License
This project is **open-source** and can be modified for personal and academic use.

---

🚀 **Feel free to contribute & improve this project!**  
**Made with ❤️ using Python & Mediapipe**  
