import json
import os
import actions  # your existing actions.py module

CONFIG_PATH = "gesture_config.json"

def load_gesture_mapping():
    if not os.path.exists(CONFIG_PATH):
        print("⚠️ No config file found, using default empty mapping.")
        return {}

    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def get_action_from_gesture(gesture, mode, mapping):
    """Returns the callable action function for a given gesture and mode."""
    try:
        action_name = mapping[mode][gesture]
        return getattr(actions, action_name)
    except KeyError:
        print(f"⚠️ No action found for gesture '{gesture}' in mode '{mode}'")
        return None
    except AttributeError:
        print(f"⚠️ Action function '{action_name}' not found in actions.py")
        return None

# Example usage:
if __name__ == "__main__":
    gesture_map = load_gesture_mapping()
    action = get_action_from_gesture("fist", "default", gesture_map)
    if action:
        action()
