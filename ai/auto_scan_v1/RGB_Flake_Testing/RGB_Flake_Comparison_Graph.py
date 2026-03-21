import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import RadioButtons

# Load JSON files
with open("true_data_points_bad.json") as f:
    true_data = json.load(f)
with open("false_data_points_bad.json") as f:
    false_data = json.load(f)

def extract_colors(data, index, max_per_image=None):
    colors = []
    for entries in data.values():
        if max_per_image and len(entries) > max_per_image:
            sampled_entries = np.random.choice(len(entries), max_per_image, replace=False)
            entries = [entries[i] for i in sampled_entries]
        for bg_rgb, flake_rgb in entries:
            colors.append([bg_rgb, flake_rgb][index])
    return np.array(colors)

# Extract colors
true_bg = extract_colors(true_data, 0)
true_flake = extract_colors(true_data, 1)
false_bg = extract_colors(false_data, 0, max_per_image=10)
false_flake = extract_colors(false_data, 1, max_per_image=750)

# Plane mapping
planes = {'RG': (0,1), 'RB': (0,2), 'BG': (1,2)}

# Default view
current_plane = 'RG'
mode = 'Normal'  # 'Normal' or 'Projection'

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

def plot_points(plane, mode):
    ax.cla()
    if mode == 'Normal':
        # Plot in full 3D
        ax.scatter(true_bg[:,0], true_bg[:,1], true_bg[:,2], c='purple', label='True BG', alpha=0.6)
        ax.scatter(true_flake[:,0], true_flake[:,1], true_flake[:,2], c='green', label='True Flake', alpha=0.8)
        ax.scatter(false_bg[:,0], false_bg[:,1], false_bg[:,2], c='purple', label='False BG', alpha=0.6)
        ax.scatter(false_flake[:,0], false_flake[:,1], false_flake[:,2], c='red', label='False Flake', alpha=0.8)
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.set_xlim(0,255)
        ax.set_ylim(0,255)
        ax.set_zlim(0,255)
    else:
        # Project onto chosen plane
        x_idx, y_idx = planes[plane]
        ax.scatter(true_bg[:,x_idx], true_bg[:,y_idx], 0, c='purple', label='True BG', alpha=0.6)
        ax.scatter(true_flake[:,x_idx], true_flake[:,y_idx], 0, c='green', label='True Flake', alpha=0.8)
        ax.scatter(false_bg[:,x_idx], false_bg[:,y_idx], 0, c='purple', label='False BG', alpha=0.6)
        ax.scatter(false_flake[:,x_idx], false_flake[:,y_idx], 0, c='red', label='False Flake', alpha=0.8)
        ax.set_xlabel(['Red','Green','Blue'][x_idx])
        ax.set_ylabel(['Red','Green','Blue'][y_idx])
        ax.set_zlabel('Projection')
        ax.set_xlim(0,255)
        ax.set_ylim(0,255)
        ax.set_zlim(0,1)
    ax.legend()
    plt.draw()

plot_points(current_plane, mode)

# Radio buttons for plane
ax_plane = plt.axes([0.02, 0.55, 0.12, 0.15], facecolor='lightgoldenrodyellow')
radio_plane = RadioButtons(ax_plane, ('RG','RB','BG'))

def change_plane(label):
    global current_plane
    current_plane = label
    plot_points(current_plane, mode)

radio_plane.on_clicked(change_plane)

# Radio buttons for mode
ax_mode = plt.axes([0.02, 0.35, 0.12, 0.15], facecolor='lightgoldenrodyellow')
radio_mode = RadioButtons(ax_mode, ('Normal','Projection'))

def change_mode(label):
    global mode
    mode = label
    plot_points(current_plane, mode)

radio_mode.on_clicked(change_mode)

plt.show()
