import json
import cv2
import os
import glob
import numpy as np

def valid_flake_data(folder=None):
    # Ask for the folder containing images
    if not folder:
        folder = input("Enter the path to the folder containing images: ").strip()

    # Gather image paths
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder, ext)))
    image_paths.sort()
    if not image_paths:
        print("No images found in the provided folder.")
        return

    saved_data = {}
    current_image_index = 0
    current_clicks = []
    original_image = None
    display_image = None

    # Zoom / pan state
    zoom_scale = 1.0
    zoom_center = None
    ZOOM_STEP = 1.25
    PAN_STEP = 50

    def draw_overlay(img, filename):
        overlay = img.copy()
        cv2.putText(overlay, filename, (10, overlay.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if filename in saved_data and saved_data[filename]:
            text = f"Saved sets: {len(saved_data[filename])}"
            cv2.putText(overlay, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return overlay

    def get_display_image():
        nonlocal zoom_center
        h, w = original_image.shape[:2]
        if zoom_center is None:
            zoom_center = (w // 2, h // 2)

        win_w, win_h = int(w / zoom_scale), int(h / zoom_scale)
        cx, cy = zoom_center
        x1 = max(0, cx - win_w // 2)
        y1 = max(0, cy - win_h // 2)
        x2 = min(w, x1 + win_w)
        y2 = min(h, y1 + win_h)

        cropped = original_image[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        disp = draw_overlay(resized, os.path.basename(image_paths[current_image_index]))
        for (x, y) in current_clicks:
            cv2.circle(disp, (x, y), 5, (0, 255, 0), -1)
        return disp

    def load_image(index):
        nonlocal original_image, display_image, current_clicks, zoom_scale, zoom_center
        path = image_paths[index]
        original_image = cv2.imread(path)
        if original_image is None:
            print("Could not load image:", path)
            return
        current_clicks = []
        zoom_scale = 1.0
        zoom_center = None
        display_image = get_display_image()
        cv2.imshow("Image", display_image)

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_clicks, display_image
        if event == cv2.EVENT_LBUTTONDOWN and len(current_clicks) < 1:
            current_clicks.append((x, y))
            cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Image", display_image)

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)
    load_image(current_image_index)

    print("""
    Instructions:
    Left-click      : Add a point (1 per set).
    c               : Clear current selection.
    s               : Save set (requires exactly 1 point).
    d / a           : Next / previous image.
    + / -           : Zoom in / out.
    ← / ↑ / → / ↓   : Pan Left / Up / Right / Down when zoomed.
    Esc             : Quit and save data.json.
    """)

    while True:
        cv2.imshow("Image", display_image)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # Esc
            break

        elif key == ord('c'):
            current_clicks = []
            display_image = get_display_image()
            print("Cleared selection.")

        elif key == ord('s'):
            if len(current_clicks) == 1:
                h, w = original_image.shape[:2]
                win_w, win_h = int(w / zoom_scale), int(h / zoom_scale)
                x_off = max(0, zoom_center[0] - win_w // 2)
                y_off = max(0, zoom_center[1] - win_h // 2)
                dx, dy = current_clicks[0]
                orig_x = int(x_off + dx / w * win_w)
                orig_y = int(y_off + dy / h * win_h)

                b, g, r = original_image[orig_y, orig_x]
                flake_color = [int(r), int(g), int(b)]
                rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                red_mode = int(np.bincount(rgb_image[:, :, 0].flatten()).argmax())
                green_mode = int(np.bincount(rgb_image[:, :, 1].flatten()).argmax())
                blue_mode = int(np.bincount(rgb_image[:, :, 2].flatten()).argmax())
                background_color = [red_mode, green_mode, blue_mode]

                rgb_set = [background_color, flake_color]
                fname = os.path.basename(image_paths[current_image_index])
                saved_data.setdefault(fname, []).append(rgb_set)
                print(f"Saved for {fname}: BG {background_color}, Flake {flake_color}")
                current_clicks = []
                display_image = get_display_image()
            else:
                print("Select exactly one point before saving.")

        elif key == ord('d'):
            if current_image_index < len(image_paths) - 1:
                current_image_index += 1
                load_image(current_image_index)
            else:
                print("Last image.")

        elif key == ord('a'):
            if current_image_index > 0:
                current_image_index -= 1
                load_image(current_image_index)
            else:
                print("First image.")

        elif key == ord('+'):
            zoom_scale = min(zoom_scale * ZOOM_STEP, 10.0)
            display_image = get_display_image()
            print(f"Zoom: {zoom_scale:.2f}×")

        elif key == ord('-'):
            zoom_scale = max(zoom_scale / ZOOM_STEP, 1.0)
            display_image = get_display_image()
            print(f"Zoom: {zoom_scale:.2f}×")

        # Pan with arrow keys (key codes 81=←, 82=↑, 83=→, 84=↓)
        elif key in (81, 82, 83, 84):
            if zoom_scale > 1.0:
                dx = PAN_STEP / zoom_scale
                cx, cy = zoom_center
                if key == 82:   cy -= dx  # Up
                elif key == 84: cy += dx  # Down
                elif key == 81: cx -= dx  # Left
                elif key == 83: cx += dx  # Right
                h, w = original_image.shape[:2]
                zoom_center = (int(np.clip(cx, 0, w)), int(np.clip(cy, 0, h)))
                display_image = get_display_image()
                print(f"Panned to {zoom_center}.")

    cv2.destroyAllWindows()

    # Save data.json next to script
    script_directory = os.getcwd()
    save_dir = os.path.join(script_directory, "datapoints")
    os.makedirs(save_dir, exist_ok=True)
    data_path = os.path.join(save_dir, "true_data_points.json")
    with open(data_path, "w") as f:
        json.dump(saved_data, f, indent=4)
    print(f"\nData saved to {data_path}")

if __name__ == "__main__":
    folder ="/Users/massimozhang/Desktop/coding/Auto_Scan1/data/TIT/10x/training_data"
    valid_flake_data(folder=folder)
