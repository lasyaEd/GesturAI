import streamlit as st
import json
import os
import inspect
import actions
import sys



CONFIG_PATH = os.getenviron["GESTURE_AI_CONFIG"]
DEFAULT_CONFIG_PATH = os.getenviron["GESTURE_AI_DEFAULT_CONFIG"]
DEFAULT_MAPPING = {}

# Load default gesture mapping
if os.path.exists(DEFAULT_CONFIG_PATH):
    with open(DEFAULT_CONFIG_PATH, "r") as f:
        DEFAULT_MAPPING = json.load(f)
else:
    print('Default mapping file not found.')
    sys.exit(-1)

mapping = DEFAULT_MAPPING.copy()
# Load current gesture mapping
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
         mapping = json.load(f)
else:
    print('Config file not found.')
    sys.exit(-1)

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