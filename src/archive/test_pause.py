import pyautogui
import time

print("⏳ You have 5 seconds to open YouTube and play a video...")
time.sleep(5)

# Step 1: Move mouse to center
screen_width, screen_height = pyautogui.size()
x = screen_width // 2
y = screen_height // 2
pyautogui.moveTo(x, y, duration=0.3)
print("🖱️ Moving to center and clicking...")
pyautogui.click()

# Step 2: Press 'k' to pause/play
time.sleep(0.5)
print("🎯 Pressing 'k' key...")
pyautogui.press("k")
