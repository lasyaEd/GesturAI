# 🖐️ GesturAI: Real-Time Hand Gesture Control System

A real-time hand gesture recognition system using MediaPipe and PyTorch to control system commands like opening a terminal, copying text, or pausing videos.

---

## 🔧 Features

- ✅ Real-time gesture recognition using webcam  
- ✅ Context-aware gesture modes (`media_mode`, `word_mode`, `presentation_mode`)  
- ✅ Customizable gesture-to-action mappings  
- ✅ Streamlit UI for editing gesture mappings  
- ✅ Cross-platform (macOS, Linux, Windows)  

---

## 📂 Project Structure

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
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Gesture Data

```bash
python collect_data.py
```

### 3. Train the Model

```bash
python training.py
```

### 4. Map Gestures to Actions

```bash
# Option A: Streamlit UI
streamlit run gesture_ui_mapper.py

# Option B: CLI
python gesture_cli_mapper.py
```

### 5. Run Gesture Controller

```bash
python run_gesture_control.py
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

## 🧠 Context Modes

- `default`
- `word_mode`
- `media_mode`
- `presentation_mode`

You can switch between modes using trigger gestures like `three_fingers`, `peace_sign`, or `ok_sign`.

---

## 🔁 Customize Your Mappings

Use `gesture_ui_mapper.py` to:

- Add new gestures  
- Map gestures to system actions  
- Delete or reset mappings  
- Save to `gesture_config.json`  

---

## 📌 Future Ideas

- Record your own custom gestures  
- Add voice feedback or sound effects  
- Integrate with specific apps (Zoom, PowerPoint, etc.)  
- Cloud-hosted UI for remote gesture mapping  

