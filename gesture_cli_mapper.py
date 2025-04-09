import json
import os
import inspect
import actions

CONFIG_PATH = "gesture_config.json"

# Load or create default mapping
def load_mapping():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {"default": {}, "media_mode": {}, "presentation_mode": {}, "word_mode": {}}

def save_mapping(mapping):
    with open(CONFIG_PATH, "w") as f:
        json.dump(mapping, f, indent=2)
    print("✅ Mapping saved to gesture_config.json\n")

def list_modes(mapping):
    print("\nAvailable modes:")
    for i, mode in enumerate(mapping):
        print(f"[{i}] {mode}")
    return list(mapping.keys())

def list_actions():
    print("\nAvailable actions:")
    functions = [name for name, obj in inspect.getmembers(actions, inspect.isfunction)
                 if not name.startswith("_")]
    for i, name in enumerate(functions):
        print(f"[{i}] {name}")
    return functions

def main():
    mapping = load_mapping()
    while True:
        print("\n🧠 Interactive Gesture Mapper")
        print("1. View current mappings")
        print("2. Add/update mapping")
        print("3. Delete a mapping")
        print("4. Save and exit")
        choice = input("Select an option: ").strip()

        if choice == "1":
            print(json.dumps(mapping, indent=2))

        elif choice == "2":
            modes = list_modes(mapping)
            mode_idx = int(input("Choose a mode: "))
            mode = modes[mode_idx]

            gesture = input("Enter gesture name (e.g., 'fist'): ").strip()
            actions_list = list_actions()
            action_idx = int(input("Select action number: "))
            action_name = actions_list[action_idx]

            mapping[mode][gesture] = action_name
            print(f"✅ Mapped '{gesture}' to '{action_name}' in '{mode}' mode")

        elif choice == "3":
            modes = list_modes(mapping)
            mode_idx = int(input("Choose a mode: "))
            mode = modes[mode_idx]
            gesture = input("Enter gesture to remove: ").strip()

            if gesture in mapping[mode]:
                del mapping[mode][gesture]
                print(f"❌ Removed '{gesture}' from '{mode}'")
            else:
                print(f"⚠️ '{gesture}' not found in '{mode}'")

        elif choice == "4":
            save_mapping(mapping)
            break

        else:
            print("⚠️ Invalid choice. Try again.")

if __name__ == "__main__":
    main()
