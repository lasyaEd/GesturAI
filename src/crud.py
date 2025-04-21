import json
import os
import src.gesture_data_capture as gesture_data_capture
import src.gesture_model as gesture_model
import pandas as pd

# TODO make sure this works correctly - MUST close the file after saving
def save_mapping(path, mapping):
    try:
        with open(path, "w") as f:
            json.dump(mapping, f, indent=2)
    except Exception as e:
        print(f"Error saving file: {e}")
        return False
    # cleanup - close file if it was opened
    finally:
        try:
            f.close()
        except NameError:
            pass
        except Exception as e:
            print(f"Error closing file: {e}")

# TODO make sure this works correctly - should close the file after loader
def load_json_mapping(path):
    try:
        with open(path, "r") as f:
            mapping = json.load(f)
        f.close()
        return mapping, True
    except FileNotFoundError:
        print(f"File not found: {path}")
        return {}, False
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {path}")
        return {}, False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}, False
    # cleanup - close file if it was opened
    finally:
        f.close()

def load_csvs(data_path):
    """
    Load training data from CSV files in the specified directory.
    
    Args:
        data_path (str): Path to the directory containing gesture data CSV files.
    
    Returns:
        pd.DataFrame: Combined DataFrame of all gesture data.
    """
    all_data = []
    for filename in os.listdir(data_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_path, filename)
            gesture_data = pd.read_csv(file_path)
            all_data.append(gesture_data)
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def save_model(model, path):
    
    # TODO move this to configuartion app
    # torch.save(model.state_dict(), "gesture_model.pth")
    # print("🎉 Model saved to gesture_model.pth")

    # # TODO import gesture labels



    # label_map = {v: k for k, v in GESTURE_LABELS.items()}

    # with open("label_map.pkl", "wb") as f:
    #     pickle.dump(label_map, f)
    # print("📌 Label map saved to label_map.pkl")

    pass


def create_context(
        context_name, 
        gesture_action_map_path):
    """
    Add a new context/mode to the gesture map.
    
    Args:
        context_name (str): Name of the new context to add
        gesture_map (dict): The current gesture action map
    
    Returns:
        bool: True if successful, False if the context already exists
    """
    # Load current gesture-action map
    gesture_action_map = load_json_mapping(gesture_action_map_path)[0]
    
    if context_name in gesture_action_map:
        return False
    
    # Add empty context to the map
    gesture_action_map[context_name] = {}
    
    # Save the updated map
    save_mapping(gesture_action_map_path, gesture_action_map)
    return True


def delete_context(
        context_name, 
        gesture_action_map_path):
    """
    Remove a context/mode from the gesture map.
    
    Args:
        context_name (str): Name of the context to remove
        gesture_map (dict): The current gesture action map
    
    Returns:
        bool: True if successful, False if the context doesn't exist or is protected
    """
    # Load current gesture-action map
    gesture_action_map = load_json_mapping(gesture_action_map_path)[0]
    
    # Protect essential contexts from being removed
    protected_contexts = ["mode_context_selection", "mode_default"]
    
    if context_name not in gesture_action_map:
        return False
    
    if context_name in protected_contexts:
        return False
    
    # Remove the context
    del gesture_action_map[context_name]
    
    # Also remove references to this context from mode_context_selection
    if "mode_context_selection" in gesture_action_map:
        context_selection = gesture_action_map["mode_context_selection"]
        # Find and remove any gesture that points to the removed context
        for gesture, action in list(context_selection.items()):
            if action == context_name:
                del context_selection[gesture]
    
    # Save the updated map
    save_mapping(gesture_action_map_path, gesture_action_map)
    return True


def update_context_name(
        current_context_name, 
        new_context_name, 
        gesture_action_map_path):
    """
    Rename a context/mode in the gesture map.
    
    Args:
        current_context_name (str): Current name of the context
        new_context_name (str): New name for the context
        gesture_map (dict): The current gesture action map
    
    Returns:
        bool: True if successful, False if the context doesn't exist, is protected, 
              or the new name already exists
    """
    # Load current gesture-action map
    gesture_action_map = load_json_mapping(gesture_action_map_path)[0]
    
    # Protect essential contexts from being renamed
    protected_contexts = ["mode_context_selection", "mode_default"]
    
    if current_context_name not in gesture_action_map:
        return False
    
    if new_context_name in gesture_action_map:
        return False
    
    if current_context_name in protected_contexts:
        return False
    
    # Create new entry with the same mappings
    gesture_action_map[new_context_name] = gesture_action_map[current_context_name]
    
    # Remove the old entry
    del gesture_action_map[current_context_name]
    
    # Update references in mode_context_selection
    if "mode_context_selection" in gesture_action_map:
        context_selection = gesture_action_map["mode_context_selection"]
        # Find and update any gesture that points to the renamed context
        for gesture, action in list(context_selection.items()):
            if action == current_context_name:
                context_selection[gesture] = new_context_name
    
    # Save the updated map
    save_mapping(gesture_action_map_path, gesture_action_map)
    return True




# CRUD operations for gesture-action mappings

def create_gesture_action_mapping(
        gesture_name, 
        action_name, 
        context_name, 
        gesture_action_map_path):
    
    new_gesture_action_map = load_json_mapping(gesture_action_map_path)[0]

    # Check if the context exists
    if context_name not in new_gesture_action_map:
        print(f"Context '{context_name}' does not exist.")
        return False
    
    # Check if the gesture already exists in the context
    if gesture_name in new_gesture_action_map[context_name]:
        print(f"Gesture '{gesture_name}' already exists in context '{context_name}'.")
        return False
    
    # Check if the action already exists in the context
    if action_name in new_gesture_action_map[context_name].values():
        print(f"Action '{action_name}' already exists in context '{context_name}'.")
        return False
    
    # Add the new gesture-action mapping to the context
    new_gesture_action_map[context_name][gesture_name] = action_name
    
    # Save the updated gesture-action map
    save_mapping(gesture_action_map_path, new_gesture_action_map)
    return True   

def delete_gesture_action_mapping(
        gesture_name, 
        action_name, 
        context_name, 
        gesture_action_map_path):
    # TODO actually write this function
    
    new_gesture_action_map = load_json_mapping(gesture_action_map_path)[0]

    # Check if the context exists
    if context_name not in new_gesture_action_map:
        print(f"Context '{context_name}' does not exist.")
        return False
    
    # Check if the gesture exists in the context
    if gesture_name not in new_gesture_action_map[context_name]:
        print(f"Gesture '{gesture_name}' does not exist in context '{context_name}'.")
        return False
    
    # Check if the gesture-action mapping exists
    if new_gesture_action_map[context_name][gesture_name] != action_name:
        print(f"Action '{action_name}' does not exist for gesture '{gesture_name}' in context '{context_name}'.")
        return False
    
    # Remove the gesture-action mapping from the context
    del new_gesture_action_map[context_name][gesture_name]

    # Save the updated gesture-action map
    save_mapping(gesture_action_map_path, new_gesture_action_map)
    return True


# CRUD operations for gestures

def create_gesture(
        gesture_name, 
        gesture_index_map_path,
        gesture_data_path,
        gesture_model_path):
    # GATHER NEW GESTURE DATA
    new_data = gesture_data_capture.gather_gesture_data(gesture_name)

    # LOAD OLD GESTURE DATA
    training_data = new_data.copy()
    training_data = pd.concat([training_data, load_csvs(gesture_data_path)], ignore_index=True)

    # CREATE NEW MAPS AND MODEL
    # TODO Create a new gesture-index map including the gesture to be added
    old_gesture_index_map = load_json_mapping(gesture_index_map_path)[0]
    new_gesture_index_map = old_gesture_index_map.copy()
    new_gesture_index_map[gesture_name] = len(old_gesture_index_map)

    # Train a new gesture model including the gesture to be added
    new_model = gesture_model.train_gesture_model(new_data)


    # UPATE MAPS AND MODEL
    # TODO LATER: Add a check to see if the model training was successful; if not, revert
    # TODO LATER: Make this an atomic operation - if any part fails, revert all changes
    # TODO Save the new gesture-index map, overwriting the old index map
    save_mapping(gesture_index_map_path, new_gesture_index_map)
    new_data.to_csv(gesture_data_path + "/" + gesture_name + ".csv", index=False)
    save_model(new_model, gesture_model_path) 
    return True


def delete_gesture(
        gesture_name, 
        gesture_index_map_path,
        gesture_action_map_path,
        gesture_data_path,
        gesture_model_path):
    
    # LOAD GESTURE DATA, DROP ROWS FROM GESTURE NAME
    training_data = pd.concat([training_data, load_csvs(gesture_data_path)], ignore_index=True)
    training_data = training_data[training_data['gesture_name'] != gesture_name]
    
    # CREATE NEW MAPS AND MODEL
    # TODO Create a new gesture-index map excluding the gesture to be removed
    old_gesture_index_map = load_json_mapping(gesture_index_map_path)[0]
    new_gesture_index_map = {}
    index = 0
    for k in old_gesture_index_map.keys():
        if k != gesture_name:
            new_gesture_index_map[k] = index
            index += 1
    
    # TODO Create a new gesture-action map excluding the gesture to be removed from all contexts
    old_gesture_action_map = load_json_mapping(gesture_action_map_path)[0]
    new_gesture_action_map = {}
    for context in old_gesture_action_map.keys():
        new_gesture_action_map[context] = {}
        for gesture in old_gesture_action_map[context].keys():
            if gesture != gesture_name:
                new_gesture_action_map[context][gesture] = old_gesture_action_map[context][gesture]
        
    # TODO Train a new gesture model excluding the gesture to be removed
    new_model = gesture_model.train_gesture_model(training_data)

    # UPDATE MAPS AND MODEL; DELETE OLD DATA
    # TODO LATER: Add a check to see if the model training was successful; if not, revert
    # TODO LATER: Make this an atomic operation - if any part fails, revert all changes
    # TODO Move save model function to CRUD & verify it works
    save_mapping(gesture_index_map_path, new_gesture_index_map)
    save_mapping(gesture_action_map_path, new_gesture_action_map)
    save_model(new_model, gesture_model_path) 
    os.delete(gesture_data_path + "/" + gesture_name + ".csv")
    return True

def update_gesture_name(
        current_gesture_name, 
        new_gesture_name,
        gesture_index_map_path,
        gesture_action_map_path,
        gesture_data_path):
    # CREATE NEW MAPS
    # Edit the name of the gesture in the gesture index map
    new_gesture_index_map = load_json_mapping(gesture_index_map_path)[0]
    if current_gesture_name not in new_gesture_index_map:
        return False
    
    new_gesture_index_map[new_gesture_name] = new_gesture_index_map[current_gesture_name]
    del new_gesture_index_map[current_gesture_name]

    # Update the gesture name in each context of the gesture map
    new_gesture_action_map = load_json_mapping(gesture_action_map_path)[0]
    for context in new_gesture_action_map.keys():
        if current_gesture_name in new_gesture_action_map[context]:
            new_gesture_action_map[context][new_gesture_name] = new_gesture_action_map[context][current_gesture_name]
            del new_gesture_action_map[context][current_gesture_name]

    # SAVE NEW MAPS & UPDATE GESTURE DATA FILE NAME
    # TODO LATER make this an atomic operation - if any part fails, revert all changes    
    save_mapping(gesture_index_map_path, new_gesture_index_map)
    save_mapping(gesture_action_map_path, new_gesture_action_map)
    os.rename(gesture_data_path + "/" + current_gesture_name + ".csv", gesture_data_path + "/" + new_gesture_name + ".csv")
    return True

# TODO add function to alter a gesture - capturing new data 
# and retraining the gesture recognition model without altering
# the gesture-action map
def update_gesture(
        gesture_name,
        gesture_index_map_path,
        gesture_data_path,
        gesture_model_path):
    # GATHER NEW GESTURE DATA
    new_data = gesture_data_capture.gather_gesture_data(gesture_name)
    
    # LOAD TRAINING DATA, REMOVE OLD GESTURE DATA, AND ADD NEW DATA
    training_data = pd.concat([training_data, load_csvs(gesture_data_path)], ignore_index=True)
    training_data = training_data[training_data['gesture_name'] != gesture_name]
    training_data = pd.concat([training_data, new_data], ignore_index=True)

    # TRAIN NEW MODEL
    # Load gesture-index map (read-only)
    gesture_index_map = load_json_mapping(gesture_index_map_path)[0]
    
    

    # TODO retrain the gesture model

    # SAVE NEW DATA AND MODEL
    # TODO LATER make this an atomic operation - if any part fails, revert all changes
    # TODO replace old gesture data with new gesture data
    # TODO replace old gesture model with new gesture model
    return True



