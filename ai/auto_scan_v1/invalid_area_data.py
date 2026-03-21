import json
import cv2
import os
import glob
import numpy as np
import itertools
from sam2_predictor import FastSAMPredictor

def invalid_area_data(folder=None):
    MAX_DISPLAY_WIDTH = 1024
    if not folder:
        folder = input("Path to images folder: ").strip()

    # find all images
    exts = ["*.jpg","*.jpeg","*.png","*.bmp","*.tiff"]
    image_paths = sorted(sum([glob.glob(os.path.join(folder, e)) for e in exts], []))
    if not image_paths:
        print("No images found."); return

    # prepare SAM
    predictor = FastSAMPredictor(
        model_cfg="configs/sam2.1/sam2.1_hiera_s.yaml",
        checkpoint="sam2.1_hiera_small.pt",
        device='cpu'  # or 'cuda'
    )

    idx = 0
    current_clicks = []    # clicks on small display
    original_image = None
    display_base = None    # downscaled BGR
    display_image = None   # overlay + clicks
    scale = 1.0
    current_background = None
    current_mask = None    # boolean mask on full-res

    def compute_bg(rgb):
        # mode per channel
        return [int(np.bincount(rgb[:,:,c].ravel()).argmax()) for c in range(3)]

    def draw_overlay(img, fname):
        o = img.copy()
        cv2.putText(o, fname, (10, o.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        # if we've already sampled true/false for this image, show count?
        return o

    def load_image(i):
        nonlocal original_image, display_base, display_image
        nonlocal scale, current_clicks, current_background, current_mask

        path = image_paths[i]
        fname = os.path.basename(path)
        original_image = cv2.imread(path)
        if original_image is None:
            raise RuntimeError("Can't load " + path)

        # compute background once
        rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        current_background = compute_bg(rgb)

        # downscale for display
        h, w = original_image.shape[:2]
        if w > MAX_DISPLAY_WIDTH:
            scale = MAX_DISPLAY_WIDTH / w
            nw, nh = int(w*scale), int(h*scale)
            display_base = cv2.resize(original_image, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            display_base = original_image.copy()

        # reset
        current_clicks = []
        current_mask = None

        # show
        predictor.set_image(original_image)
        display_image = draw_overlay(display_base, fname)
        cv2.imshow("Image", display_image)

    cv2.namedWindow("Image")

    def on_mouse(evt, x, y, flags, _):
        nonlocal current_clicks, display_image
        if evt == cv2.EVENT_LBUTTONDOWN and len(current_clicks) < 32*32:
            current_clicks.append((x, y))
            cv2.circle(display_image, (x,y), 5, (0,255,0), -1)
            cv2.imshow("Image", display_image)

    cv2.setMouseCallback("Image", on_mouse)
    load_image(idx)

    print("""
    Instructions:
    Left-click → add a point for SAM
    Space → generate mask from all clicks
    s → sample 32×32 grid, save true_points.json & false_points.json
    c → clear all clicks/mask
    d → next image
    a → prev image
    Esc → quit
    """)

    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # Esc
            break

        # clear
        elif key == ord('c'):
            load_image(idx)

        # mask
        elif key == ord(' '):
            if not current_clicks:
                print("Click at least one point.")
                continue

            # map to full-res coords
            full_pts = [(int(x/scale), int(y/scale)) for x,y in current_clicks]
            overlay_full, mask_bool = predictor.segment_points(full_pts)
            current_mask = mask_bool

            # downscale overlay to display
            dh, dw = display_base.shape[:2]
            overlay_small = cv2.resize(overlay_full, (dw, dh), interpolation=cv2.INTER_AREA)
            display_image = draw_overlay(overlay_small, os.path.basename(image_paths[idx]))
            cv2.imshow("Image", display_image)

        # sample & save
        elif key == ord('s'):
            if current_mask is None:
                print("Generate a mask first with Space.")
                continue

            H, W = current_mask.shape
            xs = np.linspace(0, W-1, 128, dtype=int)
            ys = np.linspace(0, H-1, 128, dtype=int)

            true_data = {}
            false_data = {}
            fn = os.path.basename(image_paths[idx])

            tlist, flist = [], []
            for yy, xx in itertools.product(ys, xs):
                inside = current_mask[yy, xx]
                b, g, r = original_image[yy, xx]
                flake_rgb = [int(r), int(g), int(b)]
                entry = [current_background, flake_rgb]
                if inside:
                    tlist.append(entry)
                else:
                    flist.append(entry)

            # load existing JSONs
            def load_json(p):
                return json.load(open(p)) if os.path.exists(p) else {}

            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(script_dir, "datapoints")
            os.makedirs(save_dir, exist_ok=True)
            fp = os.path.join(save_dir, "false_data_points.json")
            false_data = load_json(fp)

            true_data[fn] = tlist
            false_data[fn] = flist

            # with open(tp, "w") as f: json.dump(true_data, f, indent=4)
            with open(fp, "w") as f: json.dump(false_data, f, indent=4)

            print(f"Saved {len(flist)} false for {fn}")

        # next image
        elif key == ord('d'):
            if idx < len(image_paths)-1:
                idx += 1
                load_image(idx)
            else:
                print("Last image.")

        # prev image
        elif key == ord('a'):
            if idx > 0:
                idx -= 1
                load_image(idx)
            else:
                print("First image.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    folder =r"C:\Users\QMLab\Desktop\auto_scan\WSe2_EVE Microscope_10x - Usable"
    invalid_area_data(folder)
