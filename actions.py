import os
import webbrowser
import platform
import subprocess
from datetime import datetime
import time

OS = platform.system()

# --------------------------------------------
# System Control Functions
# --------------------------------------------

def open_terminal():
    """Opens the terminal window based on OS."""
    try:
        if OS == "Darwin":
            os.system("open -a Terminal")  # macOS
        elif OS == "Linux":
            os.system("gnome-terminal")  # Linux (change if needed)
        elif OS == "Windows":
            os.system("start cmd")
    except Exception as e:
        print(f"Error opening terminal: {e}")

def open_browser():
    """Opens the default web browser."""
    try:
        webbrowser.open("https://www.google.com")
    except Exception as e:
        print(f"Error opening browser: {e}")

def increase_volume():
    """Increases system volume based on OS."""
    try:
        if OS == "Darwin":
            os.system("osascript -e 'set volume output volume (output volume of (get volume settings) + 10)'")
        elif OS == "Linux":
            os.system("pactl set-sink-volume @DEFAULT_SINK@ +10%")
    except Exception as e:
        print(f"Error increasing volume: {e}")

def decrease_volume():
    """Decreases system volume based on OS."""
    try:
        if OS == "Darwin":
            os.system("osascript -e 'set volume output volume (output volume of (get volume settings) - 10)'")
        elif OS == "Linux":
            os.system("pactl set-sink-volume @DEFAULT_SINK@ -10%")
    except Exception as e:
        print(f"Error decreasing volume: {e}")

def take_screenshot():
    """Takes a screenshot with timestamped filename."""
    filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    try:
        if OS == "Darwin":
            os.system(f"screencapture {filename}")
        elif OS == "Linux":
            os.system(f"gnome-screenshot -f {filename}")
        elif OS == "Windows":
            os.system(f"nircmd.exe savescreenshot {filename}")  # Requires nircmd.exe in PATH
    except Exception as e:
        print(f"Error taking screenshot: {e}")

def lock_computer():
    """Locks the computer."""
    try:
        if OS == "Windows":
            os.system("rundll32.exe user32.dll,LockWorkStation")
        elif OS == "Darwin":
            os.system("/System/Library/CoreServices/Menu\\ Extras/User.menu/Contents/Resources/CGSession -suspend")
    except Exception as e:
        print(f"Error locking computer: {e}")

def open_file_explorer():
    """Opens the file explorer."""
    try:
        if OS == "Windows":
            os.system("explorer")
        elif OS == "Darwin":
            os.system("open .")
    except Exception as e:
        print(f"Error opening file explorer: {e}")

def minimize_all_windows():
    """Minimizes all windows."""
    _send_keyboard_shortcut(["win", "d"] if OS == "Windows" else ["fn", "f11"])

# --------------------------------------------
# Clipboard & Editing Functions
# --------------------------------------------

def copy():
    _send_keyboard_shortcut(["ctrl", "c"] if OS == "Windows" else ["cmd", "c"])

def paste():
    _send_keyboard_shortcut(["ctrl", "v"] if OS == "Windows" else ["cmd", "v"])

def cut():
    _send_keyboard_shortcut(["ctrl", "x"] if OS == "Windows" else ["cmd", "x"])

def undo():
    _send_keyboard_shortcut(["ctrl", "z"] if OS == "Windows" else ["cmd", "z"])

def redo():
    _send_keyboard_shortcut(["ctrl", "y"] if OS == "Windows" else ["cmd", "shift", "z"])

def select_all():
    _send_keyboard_shortcut(["ctrl", "a"] if OS == "Windows" else ["cmd", "a"])

def save():
    _send_keyboard_shortcut(["ctrl", "s"] if OS == "Windows" else ["cmd", "s"])

def print_file():
    _send_keyboard_shortcut(["ctrl", "p"] if OS == "Windows" else ["cmd", "p"])

def find():
    _send_keyboard_shortcut(["ctrl", "f"] if OS == "Windows" else ["cmd", "f"])

# --------------------------------------------
# Helper Function for Simulating Keypresses
# --------------------------------------------

def _send_keyboard_shortcut(keys):
    try:
        import pyautogui

        # macOS fix: replace 'cmd' with 'command' for pyautogui
        fixed_keys = ['command' if k == 'cmd' else k for k in keys]
        pyautogui.hotkey(*fixed_keys)

    except ImportError:
        print("pyautogui not installed. Run: pip install pyautogui")
    except Exception as e:
        print(f"Error sending keyboard shortcut {keys}: {e}")

def pause_video():
    """Click center and press Space to pause/play video."""
    try:
        import pyautogui
        screen_width, screen_height = pyautogui.size()
        x = screen_width // 2
        y = screen_height // 2
        pyautogui.moveTo(x, y, duration=0.3)
        pyautogui.click()         # Focus player
        time.sleep(0.2)
        pyautogui.press("space")  # Use spacebar instead of 'k'
        print("⏯️ Sent 'space' to pause/play video")
    except Exception as e:
        print(f"⚠️ pause_video error: {e}")




def _send_single_key(key):
    try:
        import pyautogui
        pyautogui.press(key)
    except ImportError:
        print("pyautogui not installed. Run: pip install pyautogui")
    except Exception as e:
        print(f"Error sending key '{key}': {e}")

def swipe_left_tab():
    """Three-finger swipe left = switch to previous tab (Mac)."""
    if OS == "Darwin":
        os.system("""osascript -e 'tell application "System Events" to key code 123 using control down'""")  # Left arrow

def swipe_right_tab():
    """Three-finger swipe right = switch to next tab (Mac)."""
    if OS == "Darwin":
        os.system("""osascript -e 'tell application "System Events" to key code 124 using control down'""")  # Right arrow

import pyautogui

def start_presentation():
    """Start PowerPoint presentation (equivalent to pressing F5)."""
    try:
        pyautogui.press("f5")
        print("📽 Started presentation")
    except Exception as e:
        print(f"⚠️ Error starting presentation: {e}")

def next_slide():
    """Go to the next PowerPoint slide."""
    try:
        pyautogui.press("right")
        print("➡️ Next slide")
    except Exception as e:
        print(f"⚠️ Error moving to next slide: {e}")

def previous_slide():
    """Go to the previous PowerPoint slide."""
    try:
        pyautogui.press("left")
        print("⬅️ Previous slide")
    except Exception as e:
        print(f"⚠️ Error moving to previous slide: {e}")

def exit_presentation():
    """Exits PowerPoint slideshow by bringing it to front and sending ESC."""
    try:
        import pyautogui
        import os
        import time

        # Step 1: Bring PowerPoint to the foreground
        os.system('osascript -e \'tell application "Microsoft PowerPoint" to activate\'')
        time.sleep(0.5)  # Let the window come to front

        # Step 2: Send ESC key to exit slideshow
        pyautogui.press("esc")
        print("🛑 Exited PowerPoint presentation")

    except Exception as e:
        print(f"⚠️ Error exiting PowerPoint: {e}")







# --------------------------------------------
# Gesture to Action Mapping
# --------------------------------------------

contexts = {
    "default": {
        "open_palm": paste,
        "fist": open_terminal,
        "thumbs_up": open_browser,
        "ok_sign": lock_computer,
        "thumbs_down": open_file_explorer,
        "three_fingers": None,
        "rock_on": None,
        "pinch": pause_video,
        "swipe_left": swipe_left_tab,
        "swipe_right": swipe_right_tab
    },
    "word_mode": {
        "pointing_finger": select_all,
        "open_palm": paste,
        "peace_sign": copy,
        "fist": cut,
        "thumbs_up": undo,
        "thumbs_down": redo,
        "three_fingers": None,
        "rock_on": None
    },
    "media_mode": {
        "pinch": pause_video,
        "swipe_left": decrease_volume,
        "swipe_right": increase_volume,
        "three_fingers": None,
        "rock_on": None
    }, 
    "presentation_mode": {
        "open_palm": start_presentation,
        "swipe_right": next_slide,
        "swipe_left": previous_slide,
        "three_fingers": exit_presentation
    }
}
