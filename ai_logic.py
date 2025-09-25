"""
auto_scan_pipeline.py

High‑level orchestration layer for the Auto‑Scan workflow described in
README.md:

Step 1 – Gather datapoints
Step 2 – Label datapoints
Step 3 – Train model
Step 4 – Test model on new images

All interactive logic (OpenCV/SAM UI) is delegated to the original helper
modules.

Typical usage
-------------
from auto_scan_pipeline import AutoScanPipeline

pipeline = AutoScanPipeline(
    data_folder="data/TIT/10x/training_data",
    datapoints_dir="datapoints",
    model_name="TIT_10x.h5"
)

# Step 1 (user interaction required)
pipeline.collect_valid()
pipeline.collect_invalid()

# Step 2
pipeline.label()

# Step 3
pipeline.train()

# Step 4
matrix = pipeline.test("some_image.png")
"""

import os
import json
from pathlib import Path
from typing import Optional

from ai.auto_scan.valid_flake_data import valid_flake_data
from ai.auto_scan.invalid_area_data import invalid_area_data
from ai.auto_scan.data_labeling import add_label_to_data, combine_and_shuffle
from ai.auto_scan.model import train as _train_model
from ai.auto_scan.grid_test import test_grid_batched


class AutoScanPipeline:
    """End‑to‑end workflow controller for flake selection, model training, and testing."""

    def __init__(
        self,
        data_folder: str,
        datapoints_dir: str = "datapoints",
        model_name: str = "model.h5",
    ):
        self.data_folder = Path(data_folder).expanduser().resolve()
        self.datapoints_dir = Path(datapoints_dir).expanduser().resolve()
        self.datapoints_dir.mkdir(parents=True, exist_ok=True)

        self.true_json = self.datapoints_dir / "true_data_points.json"
        self.false_json = self.datapoints_dir / "false_data_points.json"
        self.labeled_true_json = self.datapoints_dir / "labeled_true_data_points.json"
        self.labeled_false_json = self.datapoints_dir / "labeled_false_data_points.json"
        self.final_json = Path("final_data.json")
        self.model_path = Path(model_name)

    # ---------- DATA COLLECTION -------------------------------------------------

    def collect_valid(self):
        """
        Launch the OpenCV UI to collect datapoints on valid flakes
        (few‑layer, desirable thickness). Data are saved to true_data_points.json.
        """
        print("[AutoScan] Collecting VALID datapoints …")
        valid_flake_data(folder=str(self.data_folder))
        if not self.true_json.exists():
            raise RuntimeError("Expected true_data_points.json was not created.")
        print(f"[AutoScan] ✅ Saved valid datapoints → {self.true_json}")

    def collect_invalid(self):
        """
        Launch the SAM‑assisted UI to collect datapoints on invalid flakes + background.
        Data are saved to false_data_points.json.
        """
        print("[AutoScan] Collecting INVALID datapoints …")
        invalid_area_data(folder=str(self.data_folder))
        if not self.false_json.exists():
            raise RuntimeError("Expected false_data_points.json was not created.")
        print(f"[AutoScan] ✅ Saved invalid datapoints → {self.false_json}")

    # ---------- LABEL MERGING ----------------------------------------------------

    def label(self):
        """
        Convert raw datapoints to a single shuffled dataset ready for model training.
        Creates:
          labeled_true_data_points.json
          labeled_false_data_points.json
          final_data.json
        """
        print("[AutoScan] Labelling datapoints …")
        add_label_to_data(str(self.true_json), label=1, output_path=str(self.labeled_true_json))
        add_label_to_data(str(self.false_json), label=0, output_path=str(self.labeled_false_json))
        combine_and_shuffle(
            str(self.labeled_true_json),
            str(self.labeled_false_json),
            output_file=str(self.final_json),
        )
        print(f"[AutoScan] ✅ Combined & shuffled dataset → {self.final_json}")

    # ---------- TRAINING ---------------------------------------------------------

    def train(self, json_path: Optional[str] = None, **train_kwargs):
        """
        Train a small dense neural network on the labelled dataset.

        Parameters
        ----------
        json_path : str, optional
            Override default final_data.json location.
        **train_kwargs
            Extra kwargs forwarded to `model.train()` (e.g., epochs, batch_size).
        """
        json_path = json_path or str(self.final_json)
        if not Path(json_path).exists():
            raise FileNotFoundError(f"Dataset not found: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        print("[AutoScan] Training model …")
        _train_model(data, **train_kwargs)
        # `_train_model` always saves under a fixed name inside model.py,
        # but we also copy/rename for consistency
        default_output = Path("TIT_10x.h5")
        if default_output.exists():
            default_output.rename(self.model_path)
        print(f"[AutoScan] ✅ Trained model saved → {self.model_path}")

    # ---------- TESTING ----------------------------------------------------------

    def test(
        self,
        image_path: str,
        ratio: int = 14,
        batch_size: int = 4096,
        radius: int = 5,
        thickness: int = -1,
        **grid_kwargs,
    ):
        """
        Run the trained model on a new microscope image using batched grid sampling.

        Returns
        -------
        np.ndarray
            Matrix of predicted classes (shape rows × cols).
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        print(f"[AutoScan] Testing image → {image_path}")

        # grid_test expects `model` to be loaded globally; monkey‑patch here:
        import tensorflow as keras  # type: ignore
        global model
        model = keras.models.load_model(str(self.model_path))

        matrix = test_grid_batched(
            image_path,
            ratio=ratio,
            batch_size=batch_size,
            radius=radius,
            thickness=thickness,
            **grid_kwargs,
        )
        print("[AutoScan] ✅ Finished inference.")
        return matrix


if __name__ == "__main__":
    """
    Minimal CLI:
    $ python auto_scan_pipeline.py /path/to/images collect-valid
    $ python auto_scan_pipeline.py /path/to/images label
    $ python auto_scan_pipeline.py /path/to/images train
    $ python auto_scan_pipeline.py /path/to/images test /path/to/image.png
    """
    import sys

    if len(sys.argv) < 3:
        print("Usage: python auto_scan_pipeline.py <data_folder> <stage> [stage‑args…]")
        sys.exit(1)

    data_dir = sys.argv[1]
    stage = sys.argv[2]

    pipeline = AutoScanPipeline(data_dir)

    if stage == "collect-valid":
        pipeline.collect_valid()
    elif stage == "collect-invalid":
        pipeline.collect_invalid()
    elif stage == "label":
        pipeline.label()
    elif stage == "train":
        pipeline.train()
    elif stage == "test":
        if len(sys.argv) < 4:
            print("Provide an image path to test.")
            sys.exit(1)
        image = sys.argv[3]
        pipeline.test(image)
    else:
        print(f"Unknown stage: {stage}")
        sys.exit(1)
