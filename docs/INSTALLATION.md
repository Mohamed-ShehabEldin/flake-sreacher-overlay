# Installation

## Requirements

- Miniconda, Anaconda, or Miniforge with Python 3.10 or newer.
- Windows 10/11 x86-64, or macOS on Apple Silicon.
- Approximately 5 GB free for runtime setup or 10 GB for full setup.
- Internet access during setup.

The deployer uses Conda to create an environment named `flake-searcher` with
Python 3.12. It does not modify the base environment, install system-wide
packages, or request administrator access. Intel macOS is detected, but the
pinned TensorFlow and PyTorch versions do not provide Intel Mac wheels, so this
release stops with an explanation instead of attempting a source build.

## Setup

From the repository folder:

```text
python deploy_flake_searcher.py
```

The recommended **full** profile installs core UI/stage dependencies, TensorFlow
detector support, training packages, pinned SAM2, and the verified SAM2.1 small
checkpoint. The **runtime** profile omits training and SAM2 while retaining
detector inference.

Conda owns the Python environment. Small deployment state and verified download
cache files are kept under:

```text
.flake-searcher/
├── cache/
└── install-state.json
```

`environment.yml` creates the Conda environment. The selected hashed requirements
file then installs the pinned application packages. Full setup installs
PyTorch and setuptools first, downloads and verifies SAM2 source commit
`2b90b9f5ceec907a1c18123530e92e794ad901a4`, and only then builds SAM2 inside
the prepared environment. `SAM2_BUILD_CUDA=0` is set because this workflow uses
CPU segmentation and does not need SAM2's optional CUDA extension.

The checkpoint is installed atomically at
`assets/checkpoints/sam2.1_hiera_small.pt`. A matching legacy local checkpoint
is reused; otherwise it is downloaded from Meta. Its expected size is
184,416,285 bytes and its SHA-256 starts with `6d1aa6f` and ends with `5dc4d38`.

## Repeat runs and offline operation

Re-running the same setup reuses an existing compatible Python 3.12 Conda
environment, synchronizes its hashed Python packages, and reuses valid SAM2
source/checkpoint downloads. Conda resolves the base environment again only if
the dedicated environment is missing or has an incompatible Python. Invalid
downloaded files are preserved with an `.invalid` suffix before replacement.
Installation state is written only after imports, model loading, and checkpoint
checks succeed.

After successful setup, application startup does not download anything. Keep
the Conda environment, `.flake-searcher/cache/`, and the checkpoint if the
microscope must launch offline.
Creating a portable offline bundle or adding signed release-archive discovery
is a future extension point, not part of the current deployer.

## Platform notes

### macOS

Grant Screen Recording permission to the terminal or application used to launch
Flake Searcher Overlay. Accessibility permission may also be requested by
screen-control APIs. Restart the launcher after changing permissions.

### Windows microscope

The deployer does not install Arduino/USB serial drivers. Install the correct
driver for the stage controller separately if no COM port appears.

Meta currently recommends WSL for SAM2 on Windows. This project deliberately
tests a native Windows CPU route with the optional CUDA extension disabled, but
that full-profile route must still be validated on the microscope PC. Runtime
setup and TensorFlow model loading use native Windows wheels from the lock.

After setup, the deployer prints the exact `flake-searcher` interpreter path.
In VS Code press **Ctrl+Shift+P**, choose **Python: Select Interpreter**, and
select that path. Alternatively, use deployer option 4; it launches through
that exact environment interpreter and therefore cannot accidentally use the
base environment.

## Troubleshooting

- Run `python deploy_flake_searcher.py --preview` to inspect setup actions.
- Run `python deploy_flake_searcher.py --verify` after a completed setup.
- If Conda cannot be detected, rerun with
  `python deploy_flake_searcher.py --setup full --conda C:\path\to\conda.exe`.
- If an automatic SAM2 source/checkpoint download is blocked, the error shows
  the direct browser link, exact destination, and required SHA-256. Download it
  there and rerun setup; the file will be validated before use.
- If setup reports insufficient space, free space outside the repository and
  rerun; the deployer does not delete user scans, datasets, or models.
- If position becomes unknown, reconnect the stage before moving or scanning.
- If live capture is blank on macOS, recheck Screen Recording permission.
