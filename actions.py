import os
import webbrowser
import platform
from ctypes import cast, POINTER

def open_terminal():
    """Opens the terminal window based on OS."""
    try:
        if os.name == 'posix':
            os.system("open -a Terminal")  # macOS
            # os.system("gnome-terminal")  # Linux (Uncomment if using Linux)
        elif os.name == 'nt':
            os.system("start cmd")  # Windows
    except Exception as e:
        print(f"Error opening terminal: {e}")

def open_browser():
    """Opens the default web browser."""
    webbrowser.open("https://www.google.com")


def increase_volume():
    """Increases system volume based on OS."""
    system = platform.system()
    if system == "Darwin":  # macOS
        os.system("osascript -e 'set volume output volume (output volume of (get volume settings) + 10)'")
    elif system == "Linux":
        os.system("pactl set-sink-volume @DEFAULT_SINK@ +10%")

def decrease_volume():
    """Decreases system volume based on OS."""
    system = platform.system()
    if system == "Darwin":  # macOS
        os.system("osascript -e 'set volume output volume (output volume of (get volume settings) - 10)'")
    elif system == "Linux":
        os.system("pactl set-sink-volume @DEFAULT_SINK@ -10%")

def take_screenshot():
    """Takes a screenshot."""
    if os.name == "posix":
        os.system("screencapture screenshot.jpg")  # macOS
    elif os.name == "nt":
        os.system("nircmd.exe savescreenshot screenshot.png")
