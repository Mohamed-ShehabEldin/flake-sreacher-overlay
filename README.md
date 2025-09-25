
# Flake Searcher Overlay

A transparent PyQt overlay for a manual microscope. It controls a motorized XY stage, captures on-screen frames from the microscope view, and can run an external AI flake detector. This repository is a general project skeleton you can clone, run, and extend.

⚠️ Note: this repository is still in progress.

> Note: The AI model (e.g., Bo‑Zhang's) is not bundled here. Follow the model author's installation instructions and ensure it is importable in your Python environment.

---

## Features (in progrgess)
- Transparent GUI overlay on top of the microscope window
- Control of external stepper-based XY stage via serial
- Tabs for Manual control, AI inference preview, and Auto-scan
- Extensible hooks for different motor vendors and AI backends


## TODO (goals)
- [ ] **Motion bring-up**: implement `motion_controller.py` for the new XY stage (connect, move_x, move_y, get_x, get_y, homing, speeds).
- [ ] **Manual tab wiring**: hook buttons to motion methods, support press-and-hold jogging, show live position readout.
- [ ] **AI preview**: integrate Bo‑Zhang's model in `ai_logic.py`, run on screenshots from `image_frame_manager.py`, and render overlay in `ai_tab.py`.
- [ ] **Auto-scan prototype**: add serpentine raster in `autoscan_tab.py`, dwell after each move, run inference, and save detections (PNG/JPG + JSON with coords and scores).

---
