import joblib
model = joblib.load("cognitive_load_model.pkl")
from pynput import keyboard, mouse
import time
import math
import numpy as np
from pynput import keyboard, mouse
import time
import math
import numpy as np
import pandas as pd

# variables
key_count = 0
click_count = 0
mouse_positions = []

window_start = time.time()

# keyboard event
def on_press(key):
    global key_count
    key_count += 1

# mouse movement
def on_move(x, y):
    mouse_positions.append((x, y))

# mouse click
def on_click(x, y, button, pressed):
    global click_count
    if pressed:
        click_count += 1


# start listeners
keyboard_listener = keyboard.Listener(on_press=on_press)
mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)

keyboard_listener.start()
mouse_listener.start()

print("System Started... Collecting user behavior data")
print("Prediction will occur every 30 seconds")
print("Press CTRL + C to stop\n")


try:
    while True:

        time.sleep(1)

        if time.time() - window_start >= 30:

            total_time = time.time() - window_start

            # feature calculations
            keystroke_speed = key_count / total_time
            click_rate = click_count / total_time

            distance = 0
            for i in range(1, len(mouse_positions)):
                x1, y1 = mouse_positions[i-1]
                x2, y2 = mouse_positions[i]

                distance += math.sqrt((x2-x1)**2 + (y2-y1)**2)

            mouse_speed = distance / total_time if total_time > 0 else 0
            movement_variation = len(mouse_positions) / total_time if total_time > 0 else 0

            # feature vector
            features = np.array([[keystroke_speed, mouse_speed, click_rate, movement_variation]])

            # prediction
            prediction = model.predict(features)

            print("\n------ 30 Second Analysis ------")

            table = pd.DataFrame(features, columns=[
                "keystroke_speed",
                "mouse_speed",
                "click_rate",
                "movement_variation"
            ])

            print(table)

            print("\nPredicted Cognitive Load:", prediction[0])
            print("--------------------------------\n")

            # reset counters
            key_count = 0
            click_count = 0
            mouse_positions = []
            window_start = time.time()

except KeyboardInterrupt:

    print("\nSystem Stopped")

    keyboard_listener.stop()
    mouse_listener.stop()
import matplotlib.pyplot as plt
import pandas as pd
loads = ["Low","Medium","High","Medium","Low"]

mapping = {"Low":1,"Medium":2,"High":3}

numeric = [mapping[i] for i in loads]

plt.plot(numeric, marker='o')

plt.title("Cognitive Load Over Time")
plt.xlabel("Time Interval")
plt.ylabel("Load Level")

plt.yticks([1,2,3],["Low","Medium","High"])

plt.show()
import matplotlib.pyplot as plt

# Example cognitive load results collected over time
loads = ["Low", "Medium", "High", "Medium", "Low"]

# Convert labels to numbers
mapping = {"Low":1, "Medium":2, "High":3}
numeric_loads = [mapping[i] for i in loads]

# Plot graph
plt.figure()

plt.plot(numeric_loads, marker='o')

plt.title("Cognitive Load Over Time")
plt.xlabel("Time Interval (30 sec)")
plt.ylabel("Load Level")

plt.yticks([1,2,3], ["Low","Medium","High"])

plt.show()
