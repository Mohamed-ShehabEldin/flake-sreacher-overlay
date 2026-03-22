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
import sys
import json
from pathlib import Path

# add ai/auto_scan_v1 to path so we can import the student's modules directly
sys.path.insert(0, str(Path(__file__).parent / "ai" / "auto_scan_v1"))

from valid_flake_data import valid_flake_data
from invalid_area_data import invalid_area_data
from data_labeling import add_label_to_data, combine_and_shuffle
from model import train as _train_model
from grid_test import test_grid_batched


class AutoScanPipeline:

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.datapoints_dir = self.save_dir / "datapoints"
        self.datapoints_dir.mkdir(parents=True, exist_ok=True)

        self.true_json         = self.datapoints_dir / "true_data_points.json"
        self.false_json        = self.datapoints_dir / "false_data_points.json"
        self.labeled_true_json = self.datapoints_dir / "labeled_true_data_points.json"
        self.labeled_false_json= self.datapoints_dir / "labeled_false_data_points.json"
        self.final_json        = self.datapoints_dir / "final_data.json"
        self.model_path        = self.save_dir / "model.h5"

    # ---------- DATA COLLECTION -------------------------------------------------

    def collect_valid(self, folder: str):
        print("[AutoScan] Collecting VALID datapoints …")
        valid_flake_data(folder=folder, save_dir=str(self.datapoints_dir))
        print(f"[AutoScan] Saved → {self.true_json}")

    def collect_invalid(self, folder: str, checkpoint=None, max_display_width=1024, grid_sample_size=128):
        print("[AutoScan] Collecting INVALID datapoints …")
        invalid_area_data(folder=folder, save_dir=str(self.datapoints_dir), checkpoint=checkpoint,
                          max_display_width=max_display_width, grid_sample_size=grid_sample_size)
        print(f"[AutoScan] Saved → {self.false_json}")

    # ---------- LABEL MERGING ----------------------------------------------------

    def label(self):
        print("[AutoScan] Labelling datapoints …")
        add_label_to_data(str(self.true_json), label=1, output_path=str(self.labeled_true_json))
        add_label_to_data(str(self.false_json), label=0, output_path=str(self.labeled_false_json))
        combine_and_shuffle(str(self.labeled_true_json), str(self.labeled_false_json), output_file=str(self.final_json))
        print(f"[AutoScan] Saved → {self.final_json}")

    # ---------- TRAINING ---------------------------------------------------------

    def train(self, epochs=100, batch_size=32, test_size=0.2, patience=10):
        if not self.final_json.exists():
            raise FileNotFoundError(f"Run label() first — {self.final_json} not found.")
        with open(self.final_json, "r") as f:
            data = json.load(f)
        print("[AutoScan] Training model …")
        _train_model(data, epochs=epochs, batch_size=batch_size, test_size=test_size, patience=patience)
        default_output = Path("TIT_10x.h5")
        if default_output.exists():
            default_output.rename(self.model_path)
        print(f"[AutoScan] Model saved → {self.model_path}")

    # ---------- TESTING ----------------------------------------------------------

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        from tensorflow import keras as _keras
        self._model = _keras.models.load_model(str(self.model_path))
        print(f"[AutoScan] Model loaded ← {self.model_path}")

    def load_model_from_path(self, path: str):
        from tensorflow import keras as _keras
        self._model = _keras.models.load_model(path)
        print(f"[AutoScan] Model loaded ← {path}")

    def test(self, image_path, ratio=14, batch_size=4096, radius=5, thickness=-1):
        """
        Run the loaded model on an image.
        image_path: file path (str) or numpy BGR image array.
        Returns (cls_mat, img_disp).
        """
        if not hasattr(self, '_model') or self._model is None:
            raise RuntimeError("Load a model first with load_model() or load_model_from_path().")
        print(f"[AutoScan] Running inference ...")
        cls_mat, img_disp, stats = test_grid_batched(
            image_path,
            self._model,
            ratio=ratio,
            batch_size=batch_size,
            radius=radius,
            thickness=thickness,
        )
        print("[AutoScan] Finished inference.")
        return cls_mat, img_disp, stats


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
