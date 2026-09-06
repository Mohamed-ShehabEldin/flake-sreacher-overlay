# Installation

## Requirements

- Python 3.10 or newer to run the standard-library deployer.
- Windows 10/11 x86-64, or macOS on Apple Silicon.
- Approximately 5 GB free for runtime setup or 10 GB for full setup.
- Internet access during setup.

The application environment itself uses managed Python 3.12. The deployer does
not install packages globally, modify system Python, require Conda, or request
administrator access. Intel macOS is detected, but the locked TensorFlow 2.21
and PyTorch 2.10 versions do not provide Intel Mac wheels, so this release stops
with an explanation instead of attempting an unreliable source build.

Miniconda is compatible as the Python used to start the deployer, but the
application still runs in `.flake-searcher/venv`. Conda activation is not
required after setup and does not alter the locked application packages.

## Setup

From the repository folder:

```text
python deploy_flake_searcher.py
```

The recommended **full** profile installs core UI/stage dependencies, TensorFlow
detector support, training packages, pinned SAM2, and the verified SAM2.1 small
checkpoint. The **runtime** profile omits training and SAM2 while retaining
detector inference.

Managed files are kept under:

```text
.flake-searcher/
├── tools/uv[.exe]
├── python/
├── venv/
├── cache/
└── install-state.json
```

The deployer downloads uv 0.12.7 from its official GitHub release and validates
the platform archive against an embedded SHA-256 allowlist. `uv sync --locked`
then installs exactly the versions in `uv.lock`. Full setup installs SAM2 from
source commit `2b90b9f5ceec907a1c18123530e92e794ad901a4` without requiring Git and
sets `SAM2_BUILD_CUDA=0` because this workflow uses CPU segmentation.
SAM2 is built against the already locked PyTorch version rather than allowing
its isolated build metadata to fetch a different PyTorch release.

The checkpoint is installed atomically at
`assets/checkpoints/sam2.1_hiera_small.pt`. A matching legacy local checkpoint
is reused; otherwise it is downloaded from Meta. Its expected size is
184,416,285 bytes and its SHA-256 starts with `6d1aa6f` and ends with `5dc4d38`.

## Repeat runs and offline operation

Re-running the same setup reuses valid uv, Python, package cache, environment,
and checkpoint state. Invalid managed files are preserved with an `.invalid`
suffix before replacement. Installation state is written only after imports,
model loading, and checkpoint checks succeed.

After successful setup, application startup does not download anything. Keep
`.flake-searcher/` and the checkpoint if the microscope must launch offline.
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

## Troubleshooting

- Run `python deploy_flake_searcher.py --preview` to inspect setup actions.
- Run `python deploy_flake_searcher.py --verify` after a completed setup.
- If setup reports insufficient space, free space outside the repository and
  rerun; the deployer does not delete user scans, datasets, or models.
- If position becomes unknown, reconnect the stage before moving or scanning.
- If live capture is blank on macOS, recheck Screen Recording permission.
