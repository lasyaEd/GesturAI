import os

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
