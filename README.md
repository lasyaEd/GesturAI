# 🖐️ GesturAI: Real-Time Hand Gesture Control System

A real-time hand gesture recognition system using MediaPipe and PyTorch to control system commands like opening a terminal, copying text, or pausing videos.

---

## 🔧 Features

- ✅ Real-time gesture recognition using webcam  
- ✅ Context-specific gesture modes 🧠 (`media_mode`, `word_mode`, `presentation_mode`) - switch between modes using gestures!
- ✅ Customizable gesture-to-action mappings  
- ✅ Create your own gestures!  
- ✅ Streamlit UI for editing gestures and mappings  
- ✅ Cross-platform (macOS, Linux, Windows)  

---

<!-- ## 📂 Project Structure

```
GesturAI/
│
├── actions.py              # Maps gestures to system commands
├── collect_data.py         # Collects gesture data via webcam
├── gesture_data.pkl        # (ignored) Collected training data
├── gesture_model.pth       # (ignored) Trained PyTorch model
├── label_map.pkl           # (ignored) Label-to-gesture mapping
├── model.py                # PyTorch MLP model definition
├── run_gesture_control.py  # Main real-time gesture prediction script
├── training.py             # Trains the classifier
├── hand_tracker.py         # MediaPipe-based hand landmark tracking
├── gesture_ui_mapper.py    # Streamlit app for updating gesture mappings
├── gesture_cli_mapper.py   # CLI to update gesture-action mapping
├── gesture_config.json     # Saved gesture-action mappings
├── requirements.txt        # All dependencies
└── README.md               # Project overview
``` -->

---

## 🚀 Getting Started

### 1. 🧰 Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. ⚙️ Customize Your Gestures and Map Gestures to Actions

```bash
python -m streamlit run config.py
```

Use the configuration app to:
- Add, rename, relearn, or delete gestures: the app will walk you through the process of collecting data for the new gesture and automatically retrains the gesture recognition model!
- Change the way gestures trigger system actions or context changes: add or remove mappings, and add, remove, or rename contexts!

---

### 3. 🔁 Run Gesture Controller

```bash
python gesturai.py
```

---

## ✋ Example Gestures & Actions

| Gesture         | Action               | Context           |
|------------------|------------------------|--------------------|
| `fist`           | Open Terminal          | default            |
| `thumbs_up`      | Open Browser           | default            |
| `peace_sign`     | Copy                   | word_mode          |
| `palm`           | Paste                  | word_mode          |
| `pinch`          | Pause YouTube          | media_mode         |
| `swipe_right`    | Next Slide             | presentation_mode  |
| `swipe_left`     | Previous Slide         | presentation_mode  |

---

## 📌 Future Ideas

- Extend to take voice commands
- Add sound effects or feedback  
- Integrate with popular apps (Zoom, PowerPoint, etc.)  
- Cloud-hosted UI for remote gesture mapping  
