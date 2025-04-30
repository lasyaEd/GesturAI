import streamlit as st
import json
import os
import inspect
import src.actions as actions
import src.crud as crud
import sys
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

GESTURE_INDEX_MAP_PATH = os.environ["GESTURAI_GESTURE_INDEX_MAP_PATH"]
GESTURE_ACTION_MAP_PATH = os.environ["GESTURAI_GESTURE_ACTION_MAP_PATH"]
GESTURE_DATA_PATH = os.environ["GESTURAI_GESTURE_DATA_PATH"]
GESTURE_MODEL_PATH = os.environ["GESTURAI_GESTURE_MODEL_PATH"]

DEFAULT_GESTURE_INDEX_MAP_PATH = os.environ["GESTURAI_DEFAULT_GESTURE_INDEX_MAP_PATH"]
DEFAULT_GESTURE_ACTION_MAP_PATH = os.environ["GESTURAI_DEFAULT_GESTURE_ACTION_MAP_PATH"]
DEFAULT_GESTURE_DATA_PATH = os.environ["GESTURAI_DEFAULT_GESTURE_DATA_PATH"]
DEFAULT_GESTURE_MODEL_PATH = os.environ["GESTURAI_DEFAULT_GESTURE_MODEL_PATH"]


# Load current and default values for gesture-action maps and gesture-index maps
default_gesture_action_map, dga_load_success = crud.load_json_mapping(DEFAULT_GESTURE_ACTION_MAP_PATH)
if dga_load_success != True:
    print(f'Error loading default gesture action map from {DEFAULT_GESTURE_ACTION_MAP_PATH}.')
    sys.exit(-1)

gesture_action_map, ga_load_success = crud.load_json_mapping(GESTURE_ACTION_MAP_PATH)
if ga_load_success != True:
    print(f'Error loading gesture action map from {GESTURE_ACTION_MAP_PATH}.')
    sys.exit(-1)

default_gesture_index_map, dgi_load_success = crud.load_json_mapping(DEFAULT_GESTURE_INDEX_MAP_PATH)
if dgi_load_success != True:
    print(f'Error loading default gesture index map from {DEFAULT_GESTURE_INDEX_MAP_PATH}.')
    sys.exit(-1)

gesture_index_map, gi_load_success = crud.load_json_mapping(GESTURE_INDEX_MAP_PATH)
if gi_load_success != True:
    print(f'Error loading gesture index map from {GESTURE_INDEX_MAP_PATH}.')
    sys.exit(-1)

# TODO check for existence of data and model files 



# List available action functions
def list_actions():
    return [name for name, obj in inspect.getmembers(actions, inspect.isfunction)
            if not name.startswith("_")]

# UI starts here
st.set_page_config(page_title="Gesture Mapper UI", layout="centered")
st.title("🖐️ Gesture-to-Action Mapper")

modes = list(gesture_action_map.keys())
actions_list = list_actions()

st.sidebar.header("Select Mode")
selected_mode = st.sidebar.selectbox("Context Mode", modes)

st.subheader(f"Mappings in `{selected_mode}` mode")
if gesture_action_map[selected_mode]:
    for gesture, action in list(gesture_action_map[selected_mode].items()):
        col1, col2, col3 = st.columns([3, 3, 1])
        with col1:
            st.text(gesture)
        with col2:
            st.text(action)
        with col3:
            if st.button("❌", key=f"del-{selected_mode}-{gesture}"):
                # TODO fix this
                del gesture_action_map[selected_mode][gesture]
                # save_mapping(gesture_action_map)
                st.rerun()
else:
    st.info("No gestures mapped yet in this mode.")

st.markdown("---")
st.subheader("➕ Add / Update Mapping")
gesture_input = st.text_input("Gesture Name (e.g., fist, peace_sign)")
action_input = st.selectbox("Select Action", actions_list)

if st.button("Save Mapping"):
    if gesture_input:
        gesture_action_map[selected_mode][gesture_input.strip()] = action_input
        # save_mapping(gesture_action_map)
        st.success(f"Mapped '{gesture_input}' → '{action_input}' in {selected_mode} mode")
        st.rerun()
    else:
        st.warning("Please enter a gesture name.")

st.markdown("---")
if st.button("🔁 Reset to Default Mappings, Data, and Model"):
    # TODO spin this out to a crud restore-defaults function
    # TODO Confirm data and model exist before deleting
    
    # TODO Delete current gesture data, model, and mapping files
    
    # Save default gesture data, model, and mapping files as current
    crud.save_json_mapping(GESTURE_ACTION_MAP_PATH, default_gesture_action_map)
    crud.save_json_mapping(GESTURE_INDEX_MAP_PATH, default_gesture_index_map)






