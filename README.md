# Flake Searcher Overlay

A transparent PyQt5 overlay for a manual microscope. It sits on top of the microscope camera software, controls a motorized XY stage via Arduino, captures on-screen frames from the microscope view, and runs an AI flake detector to find 2D material flakes (e.g. graphene) automatically.

> ⚠️ Work in progress.

---

## Hardware Setup

- **Arduino Nano** + **2x TB6600 stepper driver** + **2x NEMA 17** (X and Y axes)
- Z axis: small stepper (planned)
- Serial baud rate: **2,000,000**

| Axis | ENA+ | DIR+ | PUL+ |
|------|------|------|------|
| X    | D13  | D11  | D12  |
| Y    | D5   | D4   | D3   |

**TB6600 DIP switches:** `OFF OFF OFF ON OFF ON` → 32 microsteps, 6400 pulses/rev, 1 A (1.2 A peak)

---

## Installation

### Step 1 — Install Miniconda

Miniconda gives you the `conda` package manager, which handles Python versions and dependencies cleanly.

**Mac:**
```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o miniconda.sh
bash miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init zsh
```
Then **restart your terminal** (or VSCode) for `conda` to be available.

**Windows:**
Download and run the installer from:
https://docs.anaconda.com/miniconda/

During install, check **"Add Miniconda to PATH"** (or use the Miniconda Prompt that gets added to your Start Menu).

---

### Step 2 — Create the environment

```bash
conda create -n flake-searcher python=3.12 -y
conda activate flake-searcher
```

> Python 3.12 is required — TensorFlow (used by the AI) does not support Python 3.13+ yet.

---

### Step 3 — Install dependencies

```bash
pip install tensorflow numpy pandas scipy opencv-python scikit-image matplotlib Pillow scikit-learn imbalanced-learn joblib torch torchvision PyQt5 pyserial pyautogui tqdm requests
```

---

### Step 3b — Install SAM2 (AI flake segmentation)

**Option A — one line (recommended):**
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

**Option B — clone first (no internet or want the source):**
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git sam2_repo
pip install sam2_repo/
```

> SAM2 installs into the conda env — no folder stays in your workspace.

---

### Step 4 — Upload the Arduino sketch

Open `stepper_ABC_ino/stepper_ABC_ino.ino` in the Arduino IDE and upload it to your Arduino Nano.

---

### Step 4b — PyTorch with GPU (Windows + NVIDIA, optional)

The `pip install` above installs CPU-only PyTorch. If you have an NVIDIA GPU and want faster SAM2 segmentation, install the CUDA version instead. Go to [pytorch.org/get-started](https://pytorch.org/get-started/locally/), select your CUDA version, and run the command shown there (e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`). Skip this if you don't have a NVIDIA GPU — CPU works fine.

---

### Step 5 — Run the app

```bash
conda activate flake-searcher
python main.py
```

**VSCode users:** After restarting VSCode, select the interpreter via `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows) → **Python: Select Interpreter** → pick `flake-searcher`.

> **macOS — Screen Recording permission:** `pyautogui` captures the screen to read the microscope view. On Mac you must grant Screen Recording permission to your Terminal (or VSCode). Go to **System Settings → Privacy & Security → Screen Recording** and enable it for Terminal / VSCode. Restart the app after granting permission.

---

## Software Structure

| File | Description |
|------|-------------|
| `main.py` | Main window — transparent, frameless, always-on-top overlay |
| `manual_tab.py` | Manual control tab — single-step and press-and-hold jogging |
| `training_ai_tab.py` | Train AI tab — collect data, set parameters, train and save a model |
| `a_eye_tab.py` | A-Eye tab — load model, run inference on an image file or live screenshot |
| `autoscan_tab.py` | Auto scan tab — serpentine grid scan with live AI inference and image saving |
| `motion_controller.py` | Serial communication with Arduino (`move_x/y/z`, `set_speed`) |
| `image_frame_manager.py` | Screenshot of the microscope view region via pyautogui |
| `ai_logic.py` | `AutoScanPipeline` — collect, label, train, and test the flake detector |
| `window_interaction_handler.py` | Drag and resize for the frameless window |
| `stepper_ABC_ino/` | Arduino sketch for TB6600 + NEMA 17 step/dir control |
| `ai/auto_scan_v1/` | AI model code — see `ai/auto_scan_v1/readme.md` for its own setup guide |

---

## Tabs

### Manual Tab
- **Single-step buttons** (`xp`/`xm`/`yp`/`ym`/`zp`/`zm`): move one step at the configured angle and speed; pressing multiple times while a move is in progress queues the extra moves — they execute one-by-one with a coordinate update after each
- **Held buttons** (`xpp`/`xmm`/`ypp`/`ymm`/`zpp`/`zmm`): hold to jog continuously, release to stop
- **Speed** and **step angle** are configurable in the tab UI
- **Connect**: select COM port and click Connect before using — on Windows ports appear as `COM3`, `COM4` etc.; on Mac as `/dev/cu.usbserial-XXXX`
- **Move to**: enter an absolute step position in the spin box next to `move_to_x/y/z` and click the button — the stage moves the exact difference from its current position

### Train AI Tab
Lets you build a new model for a new material or microscope setup:
1. Set a **save path** (where data and model will be saved)
2. Set the **SAM2 checkpoint path** (`sam2.1_hiera_small.pt`) — defaults to the one in `ai/auto_scan_v1/`
3. Click **collect valid** → select a folder of valid flake images — click flakes interactively
4. Click **collect invalid** → select a folder of images — SAM2 segments non-flake regions
5. Click **train** → labels the data and trains the dense NN
6. Click **save model** → save the resulting `.h5` file wherever you want

Configurable parameters:
| Parameter | Default | Meaning |
|-----------|---------|---------|
| max display width | 1024 | Downscale wide images during collection (display only) |
| grid sample size | 128 | Number of SAM2 sample points per axis for invalid collection |
| epochs | 100 | Max training epochs (early stopping may stop sooner) |
| batch size | 32 | Samples per gradient update |
| test split | 0.20 | Fraction of data held out for accuracy evaluation |
| early stop patience | 10 | Epochs to wait without improvement before stopping |

### A-Eye Tab
Run a trained model on a microscope image:
1. Click **Chose model** → select a `.h5` model file
2. Either:
   - **check an image** → pick an image file from disk
   - **check current window** → screenshots the microscope view region live
3. Result is shown in the panel below — green dots mark detected flake grid points
4. The **info bar** shows: `flakes: N  (raw: M)  |  HxWpx  |  bg: [R,G,B]  |  X.Xs`

Configurable grid parameters:
| Parameter | Default | Meaning |
|-----------|---------|---------|
| ratio | 5 | Grid spacing — one sample per `ratio` pixels; lower = denser |
| pred batch size | 23000 | Points sent to model at once; higher = faster |
| dot radius | 2 | Radius of drawn detection circles in pixels |

> **Note (Mac only):** `check current window` may not detect flakes because macOS applies display
> colour management when rendering images on screen, shifting pixel values by ~10/255. Loading the
> raw image file directly always works. This is expected to be fine on Windows.

---

## Auto Scan Tab

Runs a fully automated serpentine raster scan over the stage:

1. Connect the motor from the **Manual** tab first
2. Load a model in the **A-Eye** tab (optional — without a model all frames are saved)
3. In the **Auto** tab:
   - Set **fast axis** (X or Y) — the axis that sweeps quickly; the other steps between rows
   - Set **step count** and **angle** for each axis, and speeds
   - Choose **save mode**: save detected flakes only, or save all frames
   - Pick a **save folder**
   - Press **Start**

Progress is shown live: current X/Y/Z coordinates and step counter (`Step 3/100  s=0 f=2  flake=YES (47pts)`).

Saved filenames: `{x}_{y}_{z}_{yes/no}_{flake_size}.png` (auto-incremented if a file already exists).

Manual jogging buttons are also available in this tab for positioning before a scan.

---

## Current Status

### Done
- [x] Arduino sketch for TB6600 + NEMA 17 — step/dir/enable pulse control
- [x] Serial command protocol: `X {steps}`, `Y {steps}`, `S {delay_us}`
- [x] `motion_controller.py` — connect, move_x/y/z, set_speed (rev/sec)
- [x] Non-blocking motion: all serial calls run in `MotionWorker(QThread)` — GUI stays responsive
- [x] Queue-based single-step motion — rapid presses queue up and execute one-by-one
- [x] Continuous press-and-hold jogging via chained workers (hold to move, release to stop)
- [x] Manual tab fully wired — single-step and press-and-hold continuous jogging
- [x] Live position readout (X, Y, Z step counter)
- [x] Transparent frameless overlay window (always on top)
- [x] Training AI tab — full data collection, label, train, save workflow
- [x] A-Eye tab — load model, check image file or live window, display annotated result
- [x] Auto scan tab — serpentine grid scan with live AI inference, progress display, and image saving

### In Progress / Next
- [ ] Test `check current window` on Windows (expected to work — Mac colour shift issue)
- [ ] Add Z axis hardware
