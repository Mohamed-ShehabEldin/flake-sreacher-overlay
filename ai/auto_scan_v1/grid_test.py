import os
import sys
import cv2
import numpy as np
from tensorflow import keras
from tqdm import tqdm
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
    # Convert 0/1 matrix to 0/255 for OpenCV
    mat_255 = (binary_mat * 255).astype(np.uint8)
    # Label connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mat_255,
        connectivity=connectivity
    )
    # Build filtered mask
    filtered = np.zeros_like(binary_mat, dtype=np.uint8)
    for lbl in range(1, num_labels):  # skip background
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_size:
            filtered[labels == lbl] = 1
    return filtered

def test_grid_batched(image_path, model, ratio=10, radius=4, thickness=-1, batch_size=256):
    """
    Sample points on a rows×cols grid, predict in batch,
    draw green circles for flake detections, and return (cls_mat, img_disp).
    Small clusters (<5 points) are zeroed out.
    image_path can be a file path (str) or a numpy BGR image array.
    """
    start = time.time()
    if isinstance(image_path, str):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
    else:
        img_bgr = image_path
    img_disp = img_bgr.copy()
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bg_color = compute_background_color(img_rgb)

    h, w = img_rgb.shape[:2]
    print(h,w)
    rows,cols = int(h/ratio),int(w/ratio)
    dy, dx = h / rows, w / cols

    coords = []
    feats  = []
    for i in range(rows):
        for j in range(cols):
            y = int((i + 0.5) * dy)
            x = int((j + 0.5) * dx)
            coords.append((x, y))
            flake = img_rgb[y, x].tolist()
            feats.append(bg_color + flake)

    X = np.array(feats, dtype=np.float32) / 255.0

    preds = model.predict(X, batch_size=batch_size, verbose=1)
    cls   = np.argmax(preds, axis=1)

    raw_true = int(np.sum(cls == 1))

    # reshape and filter clusters
    cls_mat = cls.reshape(rows, cols)
    cls_mat = filter_clusters_by_size(cls_mat, min_size=5, connectivity=8)
    end = time.time()

    filtered_true = int(np.sum(cls_mat == 1))
    elapsed = end - start

    print(f"Image: {h}x{w}  grid: {rows}x{cols}  bg: {bg_color}")
    print(f"Raw TRUE: {raw_true}  After filter: {filtered_true}  ({elapsed:.1f}s)")

    # draw only filtered clusters
    for i in range(rows):
        for j in range(cols):
            x, y = coords[i*cols + j]
            c = cls_mat[i, j]
            color = (0,255,0) if c==1 else (0,0,0)
            cv2.circle(img_disp, (x, y), radius, color, thickness)

    stats = {
        'h': h, 'w': w,
        'rows': rows, 'cols': cols,
        'bg': bg_color,
        'raw': raw_true,
        'filtered': filtered_true,
        'elapsed': elapsed,
    }
    return cls_mat, img_disp, stats


if __name__ == '__main__':
    model_path = r"/Users/mohamedshehabeldin/Documents/GitHub/flake-sreacher-overlay/ai/auto_scan_v1/MODELS/WSe2_EVE Microscope_20x_ALPHA.h5"
    # Path to the input image
    _model = keras.models.load_model(model_path)
    test_image_path = r"/Users/mohamedshehabeldin/Documents/GitHub/flake-sreacher-overlay/ai/auto_scan_v1/WSe2_EVE Microscope_20x -  Training Data/f29_20x.png"

    matrix, img_disp, stats = test_grid_batched(test_image_path, _model, ratio=5, batch_size=4096)
    print(matrix)
    print(stats)
    cv2.imshow('result', img_disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
