# Flake Searcher Overlay

Flake Searcher Overlay is a transparent PyQt5 overlay for a manual microscope.
It controls an Arduino-driven X/Y stage, captures the microscope view, runs a
small TensorFlow flake detector, and supports normal or zigzag raster scans.

> Work in progress. Test motion with the stage clear of obstructions and keep
> the physical power/disconnect control accessible.

## Install and start

Install Miniconda, select its base Python in VS Code, and run:

```text
python deploy_flake_searcher.py
```

Choose **full installation** for detector training and SAM2, or **microscope
runtime only** for stage control, capture, Auto Scan, and detector inference.
The deployer creates or updates an isolated Conda environment named
`flake-searcher`, installs the locked packages automatically, downloads and
verifies SAM2 assets when needed, and displays the exact interpreter to select
in VS Code. Its Launch option always uses that environment.

Noninteractive commands are also available:

```text
python deploy_flake_searcher.py --setup full
python deploy_flake_searcher.py --setup runtime
python deploy_flake_searcher.py --verify
python deploy_flake_searcher.py --launch
python deploy_flake_searcher.py --setup full --preview
```

See [Installation](docs/INSTALLATION.md) for platform requirements and
troubleshooting.

## Stage behavior

- Select the serial port and connect from the Manual tab before moving.
- X/Y speed and motion commands execute as one serialized transaction.
- Software coordinates change only after an exact firmware acknowledgement.
- A timeout, malformed reply, disconnect, or serial error makes the displayed
  position unknown and aborts an active scan.
- Manual and Auto jogging are disabled while Auto Scan owns the stage; Stop and
  the safety disconnect remain available.
- Axis availability is controller-capability driven. The current firmware
  advertises X/Y only, so Z controls are disabled and no Z command is sent.

Normal and zigzag raster geometry, direction conventions, first-frame capture,
capture behavior, saving modes, and filenames are unchanged from the stabilized
application.

## Hardware

- Arduino Nano
- Two TB6600 stepper drivers
- Two NEMA 17 motors for X and Y
- Serial baud rate: 2,000,000

| Axis | ENA+ | DIR+ | PUL+ |
|---|---:|---:|---:|
| X | D13 | D11 | D12 |
| Y | D5 | D4 | D3 |

The unchanged sketch is at
`firmware/stage_controller/stage_controller.ino`.

## Project map

- `flake_searcher/`: runtime application and UI resources.
- `flake_searcher/training/`: lazily imported training and SAM2 helpers.
- `models/detector/`: the six tracked detector models.
- `firmware/stage_controller/`: current X/Y controller firmware.
- `research/`: preserved historical code, experiments, and reference data.
- `tests/`: hardware-independent stage, scan, packaging, detector, and deployer tests.

Training instructions are in [Training](docs/TRAINING.md). Development and test
commands are in [Development](docs/DEVELOPMENT.md).
