import os
import sys
import cv2
import numpy as np
from tensorflow import keras
import time


def compute_background_color(img_rgb):
    """
    Compute the background color by taking the mode (most common value)
    for each RGB channel.
    """
    background = []
    for i in range(3):
        channel = img_rgb[:, :, i].flatten()
        mode_val = int(np.bincount(channel).argmax())
        background.append(mode_val)
    return background


def filter_clusters_by_size(binary_mat, min_size=5, connectivity=8):
    """
    Remove connected-components smaller than min_size using OpenCV.
    """
    mat_255 = (binary_mat * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mat_255,
        connectivity=connectivity
    )
    filtered = np.zeros_like(binary_mat, dtype=np.uint8)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_size:
            filtered[labels == lbl] = 1
    return filtered


def test_grid_batched(image_path, ratio, radius, thickness, batch_size, min_size):
    """
    Sample points on a rows×cols grid, predict in batch,
    draw red/green circles for classes 0/1, apply cluster filtering,
    and return a matrix of class labels.
    """
    start = time.time()
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img_disp = img_bgr.copy()
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h, w = img_rgb.shape[:2]
    rows, cols = int(h / ratio), int(w / ratio)
    dy, dx = h / rows, w / cols

    bg_color = compute_background_color(img_rgb)
    coords, feats = [], []
    for i in range(rows):
        for j in range(cols):
            y = int((i + 0.5) * dy)
            x = int((j + 0.5) * dx)
            coords.append((x, y))
            flake = img_rgb[y, x].tolist()
            feats.append(bg_color + flake)

    X = np.array(feats, dtype=np.float32) / 255.0
    preds = model.predict(X, batch_size=batch_size, verbose=1)
    cls = np.argmax(preds, axis=1)

    cls_mat = cls.reshape(rows, cols)
    cls_mat = filter_clusters_by_size(cls_mat, min_size=min_size, connectivity=8)

    end = time.time()
    true_count = int(np.sum(cls_mat == 1))
    total_count = cls_mat.size
    print(f"Predicted TRUE at {true_count} out of {total_count} grid points.")
    print(f"Processing took {end - start:.2f} seconds.")

    for i in range(rows):
        for j in range(cols):
            x, y = coords[i * cols + j]
            c = cls_mat[i, j]
            color = (0, 255, 0) if c == 1 else (0, 0, 0)
            cv2.circle(img_disp, (x, y), radius, color, thickness)

    window_name = f'Batched Grid {rows}×{cols}'
    cv2.imshow(window_name, img_disp)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    return cls_mat


# === User-configurable variables ===
# Path to the trained model
model_path = "/Users/massimozhang/Desktop/coding/Auto_Scan1/TIT_10x.h5"
# Path to the input image
image_path = "/Users/massimozhang/Desktop/coding/Auto_Scan1/data/TIT/10x/training_data/1_10x.png"
# Grid sampling ratio (h/ratio × w/ratio grid)
ratio = 14
# Batch size for model prediction
batch_size = 230000
# Circle draw settings
radius = 5
thickness = -1

# Physical settings (adjust these values as needed)
# Actual width of the image in meters (set to None to disable)
width_meters = 5440  # example: 5 mm across image width
# Minimum flake area in square meters for cluster filtering (set to None to disable)
flake_area = 10   # example: 0.1 mm²

model = keras.models.load_model(model_path)

# Compute min_size from physical area if variables provided\ nmin_size = 5
if width_meters is not None and flake_area is not None:
    temp_img = cv2.imread(image_path)
    if temp_img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    h_px, w_px = temp_img.shape[:2]
    pixel_size_m = width_meters / w_px
    pixel_area_m2 = pixel_size_m ** 2
    min_size = int(flake_area / pixel_area_m2)
    print(f"Computed pixel_size: {pixel_size_m:.6f} m/pixel")
    print(f"Computed min cluster size: {min_size} grid points for area {flake_area} m^2.")
else:
    min_size = 5
    print(f"Using default min cluster size: {min_size} grid points.")

# Run detection
matrix = test_grid_batched(
    image_path,
    ratio=ratio,
    radius=radius,
    thickness=thickness,
    batch_size=batch_size,
    min_size=min_size
)
print(matrix)
