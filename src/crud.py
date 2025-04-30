import json
import os
import src.gesture_data as gd
import src.gesture_model as gm
import pandas as pd
import torch

# SUPPORT FUNCTIONS - 
# TODO consider removing these and using a library like `jsonschema` for validation

def save_json_mapping(path, mapping):
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

def load_json_mapping(path):
    try:
        with open(path, "r") as f:
            mapping = json.load(f)
        f.close()
        return mapping
    except FileNotFoundError:
        print(f"File not found: {path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {path}")
        return {}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}
    # cleanup - close file if it was opened
    finally:
        f.close()

def load_gesture_data(
        data_path, 
        gesture_name, 
        suffix = ".csv"):
    """
    Load training data from a CSV file.
    
    Args:
        data_path (str): Path to the CSV file containing gesture data.
    
    Returns:
        pd.DataFrame: DataFrame containing the gesture data.
    """
    file_path = os.path.join(data_path, gesture_name + suffix)
    try:
        gesture_data = pd.read_csv(file_path)
        # add a column 'gesture_name' to the data
        gesture_data['gesture_name'] = gesture_name
        return gesture_data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Empty CSV file: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error: {e}")
        return pd.DataFrame()


# CONTEXT CRUD OPERATIONS

def create_context(
        context_name, 
        gesture_action_map_path):
    """
    Add a new context/mode to the gesture map.
    
    Args:
        context_name (str): Name of the new context to add
        gesture_map (dict): The current gesture action map
    
    Returns:
        bool: True if successful, False if the context already exists or if the name is invalid
    """
    if context_name[:5] != "mode_":
        print(f"Context name '{context_name}' must start with 'mode_'.")
        return False

    # Load current gesture-action map
    gesture_action_map = load_json_mapping(gesture_action_map_path)
    
    if context_name in gesture_action_map:
        return False
    
    # Add empty context to the map
    gesture_action_map[context_name] = {}
    
    # Save the updated map
    save_json_mapping(gesture_action_map_path, gesture_action_map)
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
    gesture_action_map = load_json_mapping(gesture_action_map_path)
    
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
    save_json_mapping(gesture_action_map_path, gesture_action_map)
    return True

def rename_context(
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
    gesture_action_map = load_json_mapping(gesture_action_map_path)
    
    # Protect essential contexts from being renamed
    protected_contexts = ["mode_context_selection"]
    if current_context_name in protected_contexts:
        print(f"Cannot rename protected context: {current_context_name}")
        return False

    if current_context_name not in gesture_action_map.keys():
        print(f"Context '{current_context_name}' does not exist.")
        return False
    if new_context_name in gesture_action_map.keys():
        print(f"Context '{new_context_name}' already exists.")
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
    save_json_mapping(gesture_action_map_path, gesture_action_map)
    return True



# GESTURE-ACTION CRUD OPERATIONS

def create_gesture_action_mapping(
        gesture_name, 
        action_name, 
        context_name, 
        gesture_action_map_path):
    
    gesture_action_map = load_json_mapping(gesture_action_map_path)

    # Check if the context exists
    if context_name not in gesture_action_map:
        print(f"Context '{context_name}' does not exist.")
        return False
    
    # Check if the gesture already exists in the context
    if gesture_name in gesture_action_map[context_name]:
        print(f"Gesture '{gesture_name}' already exists in context '{context_name}'.")
        return False
    
    # Check if the action already exists in the context
    if action_name in gesture_action_map[context_name].values():
        print(f"Action '{action_name}' already exists in context '{context_name}'.")
        return False
    
    # Add the new gesture-action mapping to the context
    gesture_action_map[context_name][gesture_name] = action_name
    
    # Save the updated gesture-action map
    save_json_mapping(gesture_action_map_path, gesture_action_map)
    return True   

def delete_gesture_action_mapping(
        gesture_name, 
        action_name, 
        context_name, 
        gesture_action_map_path):
    
    gesture_action_map = load_json_mapping(gesture_action_map_path)

    # Check if the context exists
    if context_name not in gesture_action_map:
        print(f"Context '{context_name}' does not exist.")
        return False
    
    # Check if the gesture exists in the context
    if gesture_name not in gesture_action_map[context_name]:
        print(f"Gesture '{gesture_name}' does not exist in context '{context_name}'.")
        return False
    
    # Check if the gesture-action mapping exists
    if gesture_action_map[context_name][gesture_name] != action_name:
        print(f"Action '{action_name}' does not exist for gesture '{gesture_name}' in context '{context_name}'.")
        return False
    
    # Remove the gesture-action mapping from the context
    del gesture_action_map[context_name][gesture_name]

    # Save the updated gesture-action map
    save_json_mapping(gesture_action_map_path, gesture_action_map)
    return True


# GESTURE CRUD OPERATIONS

def create_gesture(
        created_gesture_name, 
        gesture_index_map_path,
        gesture_data_path,
        gesture_model_path):
    
    # LOAD GESTURE INDEX MAP AND ADD NEW GESTURE-INDEX ITEM
    gesture_index_map = load_json_mapping(gesture_index_map_path)
    gesture_index_map[created_gesture_name] = len(gesture_index_map)

    # LOAD EXISTING GESTURE DATA
    training_dfs = []
    
    for gesture_name in gesture_index_map.keys():
        # Load the gesture data and add a column for the gesture name
        if gesture_name != created_gesture_name:
            gesture_data = load_gesture_data(
                gesture_data_path, 
                gesture_name, 
                ".csv")
            training_dfs.append(gesture_data)
               
    # GATHER NEW GESTURE DATA
    new_data = gd.gather_gesture_data(created_gesture_name)
    training_dfs.append(new_data)

    # CONCATENATE ALL TRAINING DATA
    training_data = pd.concat(training_dfs, ignore_index=True)

    # TRAIN NEW MODEL
    model = gm.model_training_pipeline(
        training_data, 
        gesture_index_map)

    # SAVE UPDATED MAPS AND MODEL
    # TODO LATER: Add a check to see if the model training was successful; if not, revert
    # TODO LATER: Make this an atomic operation - if any part fails, revert all changes
    
    # Save the new gesture-index map, overwriting the old index map
    save_json_mapping(gesture_index_map_path, gesture_index_map)
    
    # Save the data for the new gesture
    new_data = new_data.drop(columns=['gesture_name'])
    new_data.to_csv(gesture_data_path + created_gesture_name + ".csv", index=False)
    
    #Save the new model, overwriting the old model
    torch.save(model, gesture_model_path)
    
    return True




# Add function to alter a gesture - capturing new data 
# and retraining the gesture recognition model without altering
# the gesture-action map
def retrain_gesture(
        retrained_gesture_name,
        gesture_index_map_path,
        gesture_data_path,
        gesture_model_path):
    
    # LOAD GESTURE INDEX MAP (READ-ONLY)
    gesture_index_map = load_json_mapping(gesture_index_map_path)
    
    # LOAD TRAINING DATA, REPLACING OLD GESTURE DATA WITH NEW GESTURE DATA
    training_dfs = []
    
    for gesture_name in gesture_index_map.keys():
        if gesture_name != retrained_gesture_name:
            # Load the gesture data and add a column for the gesture name
            gesture_data = load_gesture_data(
                gesture_data_path, 
                gesture_name,  
                ".csv")
            training_dfs.append(gesture_data)
        else:
            # GATHER NEW GESTURE DATA
            new_gesture_data = gd.gather_gesture_data(retrained_gesture_name)
            training_dfs.append(new_gesture_data)

    # CONCATENATE ALL TRAINING DATA
    training_data = pd.concat(training_dfs, ignore_index=True)

    # TRAIN NEW MODEL
    model = gm.model_training_pipeline(training_data, gesture_index_map)

    # SAVE NEW DATA AND MODEL
    # TODO LATER make this an atomic operation - if any part fails, revert all changes
    # replace old gesture data with new gesture data
    new_gesture_data = new_gesture_data.drop(columns=['gesture_name'])
    new_gesture_data.to_csv(gesture_data_path + "/" + retrained_gesture_name + ".csv", index=False)
    
    #Save the new model, overwriting the old model
    torch.save(model, gesture_model_path)

    return True

# TODO problem - doesn't rename the gesture within the data file - but this is extraneous
# TODO refactor project to exclude score, capture_number, and gesture_name from the data file
def rename_gesture(
        current_gesture_name, 
        new_gesture_name,
        gesture_index_map_path,
        gesture_action_map_path,
        gesture_data_path):
    # CREATE NEW MAPS
    # Update the gesture name in the gesture-index map
    gesture_index_map = load_json_mapping(gesture_index_map_path)
    gesture_index_map[new_gesture_name] = gesture_index_map[current_gesture_name]
    del gesture_index_map[current_gesture_name]

    # Update the gesture name in each context of the gesture-action map
    gesture_action_map = load_json_mapping(gesture_action_map_path)
    for context in gesture_action_map.keys():
        if current_gesture_name in gesture_action_map[context]:
            gesture_action_map[context][new_gesture_name] = gesture_action_map[context][current_gesture_name]
            del gesture_action_map[context][current_gesture_name]

    # SAVE NEW MAPS & UPDATE GESTURE DATA FILE NAME
    # TODO LATER make this an atomic operation - if any part fails, revert all changes    
    save_json_mapping(gesture_index_map_path, gesture_index_map)
    save_json_mapping(gesture_action_map_path, gesture_action_map)
    os.rename(gesture_data_path + "/" + current_gesture_name + ".csv", gesture_data_path + "/" + new_gesture_name + ".csv")
    return True



def delete_gesture(
        deleted_gesture_name, 
        gesture_index_map_path,
        gesture_action_map_path,
        gesture_data_path,
        gesture_model_path):
    
    # LOAD AND UPDATE MAPS TO EXCLUDE TARGET GESTURE 
    gesture_index_map = load_json_mapping(gesture_index_map_path)
    # Remove the gesture from the index map
    del gesture_index_map[deleted_gesture_name]
    # Renumber the indices of the remaining gestures
    i = 0
    new_gesture_index_map = {}
    for k in gesture_index_map.keys():
        new_gesture_index_map[k] = i
        i += 1
    gesture_index_map = new_gesture_index_map

    print(f"Gesture index map: {gesture_index_map}")

    # Remove the gesture from each context of the gesture-action map
    gesture_action_map = load_json_mapping(gesture_action_map_path)
    for v in gesture_action_map.values():
        if deleted_gesture_name in v.keys():
            del v[deleted_gesture_name]

    print (f"Gesture action map: {gesture_action_map}")

    # LOAD TRAINING DATA - EXCLUDES TARGET GESTURE DATA
    training_dfs = []
    
    for gesture_name in gesture_index_map.keys():
        # Load the gesture data and add a column for the gesture name
        gesture_data = load_gesture_data(
            gesture_data_path, 
            gesture_name, 
            ".csv")
        training_dfs.append(gesture_data)

    # CONCATENATE ALL TRAINING DATA
    training_data = pd.concat(training_dfs, ignore_index=True)
    
    print(f"Training data: {training_data}")
    print(f"Training data shape: {training_data.shape}")

    # TRAIN NEW MODEL
    model = gm.model_training_pipeline(
        training_data, 
        gesture_index_map)

    # SAVE MAPS AND MODEL; DELETE OLD DATA
    # TODO LATER: Add a check to see if the model training was successful; if not, revert
    # TODO LATER: Make this an atomic operation - if any part fails, revert all changes
    
    # Save the new mappigns, overwriting the old mappings
    save_json_mapping(gesture_index_map_path, gesture_index_map)
    save_json_mapping(gesture_action_map_path, gesture_action_map)
    
    #Save the new model, overwriting the old model
    torch.save(model, gesture_model_path)
    
    #Delete the old gesture data file
    os.remove(gesture_data_path + "/" + deleted_gesture_name + ".csv")
    return True