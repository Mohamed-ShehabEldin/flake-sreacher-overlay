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

## Software Structure

| File | Description |
|------|-------------|
| `main.py` | Main window — transparent, frameless, always-on-top overlay |
| `manual_tab.py` | Manual control tab — single-step and press-and-hold jogging |
| `ai_tab.py` | AI settings tab |
| `autoscan_tab.py` | Auto scan tab |
| `motion_controller.py` | Serial communication with Arduino (`move_x/y/z`, `set_speed`) |
| `image_frame_manager.py` | Screenshot of the microscope view region via pyautogui |
| `ai_logic.py` | `AutoScanPipeline` — collect, label, train, and test the flake detector |
| `window_interaction_handler.py` | Drag and resize for the frameless window |
| `stepper_ABC_ino/` | Arduino sketch for TB6600 + NEMA 17 step/dir control |
| `ai/auto_scan_v1/` | AI model code using SAM2 for flake segmentation |

---

## Current Status

### Done
- [x] Arduino sketch for TB6600 + NEMA 17 — step/dir/enable pulse control
- [x] Serial command protocol: `X {steps}`, `Y {steps}`, `S {delay_us}`
- [x] `motion_controller.py` — connect, move_x/y/z, set_speed (rev/sec)
- [x] Manual tab fully wired — single-step buttons and press-and-hold continuous jogging (xpp/ypp/zpp)
- [x] Live position readout (X, Y, Z step counter)
- [x] Transparent frameless overlay window (always on top)

### In Progress / Next
- [ ] Wire `autoscan_tab.py` to `AutoScanPipeline` for real-time grid scanning
- [ ] Fix `image_frame_manager.py` screenshot capture
- [ ] Implement AI check in manual tab (single-frame inference)
- [ ] Add Z axis hardware
- [ ] Serpentine raster scan with dwell + inference + save detections (PNG/JSON)

---

## AI Pipeline

The AI model (`ai/auto_scan_v1/`) uses **SAM2** (Segment Anything Model 2) for flake segmentation, combined with a small dense neural network trained on collected microscope images.

Steps:
1. Collect valid/invalid flake datapoints interactively
2. Label and combine into a dataset
3. Train the model (`model.h5`)
4. Run grid inference on new microscope images

> The SAM2 model weights are not bundled. Follow [SAM2 installation instructions](https://github.com/facebookresearch/segment-anything-2) and ensure it is importable in your Python environment.

---

## Requirements

```
pip install -r requirements.txt
```
