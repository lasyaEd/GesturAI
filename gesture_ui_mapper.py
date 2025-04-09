import streamlit as st
import json
import os
import inspect
import actions

CONFIG_PATH = "gesture_config.json"
DEFAULT_MAPPING = {
  "default": {
    "open_palm": "paste",
    "fist": "open_terminal",
    "thumbs_up": "open_browser",
    "ok_sign": "lock_computer",
    "thumbs_down": "open_file_explorer",
    "three_fingers": "None",
    "rock_on": "None",
    "pinch": "pause_video",
    "swipe_left": "swipe_left_tab",
    "swipe_right": "swipe_right_tab"
  },
  "word_mode": {
    "pointing_finger": "select_all",
    "open_palm": "paste",
    "peace_sign": "copy",
    "fist": "cut",
    "thumbs_up": "undo",
    "thumbs_down": "redo",
    "three_fingers": "None",
    "rock_on": "None"
  },
  "media_mode": {
    "swipe_left": "decrease_volume",
    "swipe_right": "increase_volume",
    "three_fingers": "None",
    "rock_on": "None",
    "pinch": "pause_video",
    "open_palm": "pause_video"
  },
  "presentation_mode": {
    "open_palm": "start_presentation",
    "swipe_right": "next_slide",
    "swipe_left": "previous_slide",
    "three_fingers": "exit_presentation"
  }
}

# Utility to load and save mapping
def load_mapping():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return DEFAULT_MAPPING.copy()

def save_mapping(mapping):
    with open(CONFIG_PATH, "w") as f:
        json.dump(mapping, f, indent=2)

# List available action functions
def list_actions():
    return [name for name, obj in inspect.getmembers(actions, inspect.isfunction)
            if not name.startswith("_")]

# UI starts here
st.set_page_config(page_title="Gesture Mapper UI", layout="centered")
st.title("🖐️ Gesture-to-Action Mapper")

mapping = load_mapping()
modes = list(mapping.keys())
actions_list = list_actions()

st.sidebar.header("Select Mode")
selected_mode = st.sidebar.selectbox("Context Mode", modes)

st.subheader(f"Mappings in `{selected_mode}` mode")
if mapping[selected_mode]:
    for gesture, action in list(mapping[selected_mode].items()):
        col1, col2, col3 = st.columns([3, 3, 1])
        with col1:
            st.text(gesture)
        with col2:
            st.text(action)
        with col3:
            if st.button("❌", key=f"del-{selected_mode}-{gesture}"):
                del mapping[selected_mode][gesture]
                save_mapping(mapping)
                st.rerun()
else:
    st.info("No gestures mapped yet in this mode.")

st.markdown("---")
st.subheader("➕ Add / Update Mapping")
gesture_input = st.text_input("Gesture Name (e.g., fist, peace_sign)")
action_input = st.selectbox("Select Action", actions_list)

if st.button("Save Mapping"):
    if gesture_input:
        mapping[selected_mode][gesture_input.strip()] = action_input
        save_mapping(mapping)
        st.success(f"Mapped '{gesture_input}' → '{action_input}' in {selected_mode} mode")
        st.rerun()
    else:
        st.warning("Please enter a gesture name.")

st.markdown("---")
if st.button("🔁 Reset to Default Mappings"):
    save_mapping(DEFAULT_MAPPING.copy())
    st.success("Mappings reset to default.")
    st.rerun()