# Auto Scan v1 — AI Flake Detector

Detects valid 2D material flakes (e.g. graphene, WSe2) in microscope images using SAM2 for segmentation and a small dense neural network for classification.

The pipeline has two modes:
- **Standalone** — run the scripts directly from this folder
- **Integrated** — used via the Flake Searcher Overlay app (Train AI tab + A-Eye tab)

---

## Installation

### Step 1 — Install Miniconda

**Mac (Apple Silicon):**
```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o miniconda.sh
bash miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init zsh
```
Restart your terminal after this.

**Windows:**
Download and run the installer from https://docs.anaconda.com/miniconda/
Check **"Add Miniconda to PATH"** during install, then use the Miniconda Prompt.

---

### Step 2 — Create the environment

```bash
conda create -n flake-searcher python=3.12 -y
conda activate flake-searcher
```

> Python 3.12 is required — TensorFlow does not support Python 3.13+ yet.

---

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

Or from the repo root:
```bash
pip install tensorflow numpy pandas scipy opencv-python scikit-image matplotlib Pillow scikit-learn imbalanced-learn joblib torch torchvision PyQt5 pyserial pyautogui tqdm requests
```

---

### Step 4 — Install SAM2

**Option A — one line, no folder needed (recommended):**
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

**Option B — clone first then install (if you have no internet or want the source):**
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git sam2_repo
pip install sam2_repo/
```

> SAM2 gets installed into the conda env. You do NOT need to keep `sam2_repo/` in your workspace after installing.

---

### Step 5 — Download the SAM2 checkpoint

Place the checkpoint file at `ai/auto_scan_v1/sam2.1_hiera_small.pt` (the app defaults to this path).

**Mac / Linux:**
```bash
curl -L -o ai/auto_scan_v1/sam2.1_hiera_small.pt \
     https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

**Windows:**
```bash
curl -L -o ai\auto_scan_v1\sam2.1_hiera_small.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

Or download manually from the link above.

---

## Usage — Standalone Scripts

### 1. Collect valid flake datapoints

```bash
python valid_flake_data.py
```

Set `folder` and `save_dir` at the top of the script. An OpenCV window opens for each image.

**Controls:**
- `Left click` — place a point on a valid flake
- `S` — save the current point
- `C` — clear the current point
- `A` / `D` — previous / next image
- `+` / `-` — zoom in / out
- Arrow keys — pan when zoomed in
- `ESC` — quit

Saves `datapoints/true_data_points.json`.

---

### 2. Collect invalid flake and background datapoints

```bash
python invalid_area_data.py
```

Set `folder`, `save_dir`, and `checkpoint` (path to `sam2.1_hiera_small.pt`) at the top.

**Controls:**
- `Left click` — add a point to guide SAM2 segmentation
- `Space` — generate segmentation mask around valid flakes
- `S` — save datapoints once the mask looks good
- `C` — clear all clicks and masks, restart current image
- `A` / `D` — previous / next image
- `ESC` — quit

> Make sure all valid flake regions are covered by the mask before saving. The points OUTSIDE the mask become the invalid datapoints.

Saves `datapoints/false_data_points.json`.

---

### 3. Label and combine the data

```bash
python data_labeling.py
```

Reads `true_data_points.json` and `false_data_points.json`, adds class labels, combines and shuffles them.
Outputs `datapoints/final_data.json`.

---

### 4. Train the model

```bash
python model.py
```

Trains a small dense neural network on `final_data.json`.
Outputs `model.h5` (or path configured in script).

Configurable at the top of `model.py`:
| Parameter | Default | Meaning |
|-----------|---------|---------|
| epochs | 100 | Max training epochs |
| batch_size | 32 | Samples per gradient update |
| test_size | 0.20 | Fraction held out for evaluation |
| patience | 10 | Early stopping patience |

---

### 5. Test on a new image

```bash
python grid_test.py
```

Set `model_path`, `test_image_path`, `ratio`, `batch_size`, `radius` at the bottom of the script.

```python
if __name__ == '__main__':
    model_path = r"path/to/model.h5"
    test_image_path = r"path/to/image.png"
    _model = keras.models.load_model(model_path)
    matrix, img_disp, stats = test_grid_batched(test_image_path, _model, ratio=5, batch_size=4096)
    cv2.imshow('result', img_disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

`test_grid_batched` returns `(cls_mat, img_disp, stats)`:
- `cls_mat` — 2D matrix of predicted classes (0=background, 1=flake, 2=other)
- `img_disp` — original image annotated with green circles on detected flakes
- `stats` — dict with `h, w, rows, cols, bg, raw, filtered, elapsed`

> Note: scratches and bilayers can produce false positives.

---

## Usage — Via the Overlay App

The **Train AI** tab in the app replaces steps 1–4 above with a GUI:
- Set save path and SAM2 checkpoint path
- Collect valid / invalid interactively
- Click train → click save model

The **A-Eye** tab replaces step 5:
- Load a `.h5` model
- Click **check an image** (from disk) or **check current window** (live screenshot)
- Result shown in the panel with stats

### Recommended parameters by magnification

| Magnification | ratio | dot radius |
|---------------|-------|------------|
| 50x           | 10    | 4          |
| 20x           | 5     | 2          |
| 10x           | 3     | 1          |

---

## Model details

- **Input**: 6 features per grid point — `[bg_R, bg_G, bg_B, px_R, px_G, px_B]` (normalised to 0–1)
  - `bg` = background colour (mode of each channel across the whole image)
  - `px` = colour at the sampled grid point
- **Output**: 3 classes — 0 background, 1 valid flake, 2 other (e.g. thick/bilayer)
- **Architecture**: small dense neural network (TensorFlow/Keras), saved as `.h5`
- **Post-processing**: connected-component cluster filter removes isolated detections < 5 grid points

### Known limitation — display colour shift (Mac)

The model is trained on raw image files. On Mac, the OS applies ICC colour management when displaying images, shifting pixel values by ~10/255 per channel. This means the `check current window` (screenshot) path may fail on Mac while the raw file path works fine.
On Windows this effect is much smaller and inference from screenshots is expected to work normally.

---

## Training data format

Images can be: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

Place all training images in a single folder and point the scripts (or the app) to it.
