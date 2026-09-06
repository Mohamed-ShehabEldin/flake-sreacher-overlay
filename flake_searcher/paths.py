"""Stable paths for packaged application resources."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
UI_ROOT = PACKAGE_ROOT / "ui"
MODELS_ROOT = PROJECT_ROOT / "models" / "detector"
ASSETS_ROOT = PROJECT_ROOT / "assets"
CHECKPOINTS_ROOT = ASSETS_ROOT / "checkpoints"
DEFAULT_SAM2_CHECKPOINT = CHECKPOINTS_ROOT / "sam2.1_hiera_small.pt"
FIRMWARE_ROOT = PROJECT_ROOT / "firmware" / "stage_controller"


def ui_path(filename: str) -> Path:
    """Return an absolute path to a bundled Qt Designer file."""
    path = UI_ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(f"UI resource not found: {path}")
    return path
