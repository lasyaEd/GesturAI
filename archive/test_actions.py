import time
from actions import (
    copy, paste, cut, undo, redo,
    select_all, save, print_file, find,
    open_terminal, open_browser, take_screenshot,
    lock_computer, open_file_explorer, minimize_all_windows,
    increase_volume, decrease_volume, pause_video
)

def test_shortcuts():
    print("Testing in 3 seconds... Switch to a text editor or browser window now!")
    time.sleep(3)

    print("Select All (e.g., select everything in a text editor)")
    select_all()
    time.sleep(1)

    print("Copy")
    copy()
    time.sleep(1)

    print("Paste")
    paste()
    time.sleep(1)

    print("Cut")
    cut()
    time.sleep(1)

    print("Undo")
    undo()
    time.sleep(1)

    print("Redo")
    redo()
    time.sleep(1)

    print("Save")
    save()
    time.sleep(1)

    print("Print")
    print_file()
    time.sleep(1)

    print("Find")
    find()
    time.sleep(1)

def test_system_commands():
    print("Opening Terminal...")
    open_terminal()
    time.sleep(2)

    print("Opening Browser...")
    open_browser()
    time.sleep(2)

    print("Taking Screenshot...")
    take_screenshot()
    time.sleep(2)

    print("Opening File Explorer...")
    open_file_explorer()
    time.sleep(2)

    print("Minimizing All Windows...")
    minimize_all_windows()
    time.sleep(2)

def test_volume_controls():
    print("Increasing Volume...")
    increase_volume()
    time.sleep(1)

    print("Decreasing Volume...")
    decrease_volume()
    time.sleep(1)

def test_pause_video():
    print("Switch to YouTube or VLC... Pausing/Playing in 3 seconds!")
    time.sleep(3)
    pause_video()

# Comment/uncomment what you want to test:
if __name__ == "__main__":
    #test_shortcuts()
    #test_system_commands()
    #test_volume_controls()
    test_pause_video
