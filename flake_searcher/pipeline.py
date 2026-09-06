"""Orchestration for data collection, training, and detector inference.

Heavy detector and training dependencies are imported only by the operation that
needs them. Opening the application therefore does not import TensorFlow,
PyTorch, scikit-learn, or SAM2.
"""

import json
from pathlib import Path


class AutoScanPipeline:
    def __init__(self, save_dir: str | None = None):
        self._model = None
        self.save_dir = None
        self.datapoints_dir = None
        self.true_json = None
        self.false_json = None
        self.labeled_true_json = None
        self.labeled_false_json = None
        self.final_json = None
        self.model_path = None

        if save_dir is not None:
            self.configure_workspace(save_dir)

    def configure_workspace(self, save_dir: str) -> None:
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.datapoints_dir = self.save_dir / "datapoints"
        self.datapoints_dir.mkdir(parents=True, exist_ok=True)
        self.true_json = self.datapoints_dir / "true_data_points.json"
        self.false_json = self.datapoints_dir / "false_data_points.json"
        self.labeled_true_json = self.datapoints_dir / "labeled_true_data_points.json"
        self.labeled_false_json = self.datapoints_dir / "labeled_false_data_points.json"
        self.final_json = self.datapoints_dir / "final_data.json"
        self.model_path = self.save_dir / "model.h5"

    def _require_workspace(self) -> None:
        if self.save_dir is None:
            raise RuntimeError("Choose a training save directory first.")

    def collect_valid(self, folder: str):
        self._require_workspace()
        from .training.valid_collection import valid_flake_data

        print("[AutoScan] Collecting VALID datapoints …")
        valid_flake_data(folder=folder, save_dir=str(self.datapoints_dir))
        print(f"[AutoScan] Saved → {self.true_json}")

    def collect_invalid(
        self,
        folder: str,
        checkpoint=None,
        max_display_width=1024,
        grid_sample_size=128,
    ):
        self._require_workspace()
        from .training.invalid_collection import invalid_area_data

        print("[AutoScan] Collecting INVALID datapoints …")
        invalid_area_data(
            folder=folder,
            save_dir=str(self.datapoints_dir),
            checkpoint=checkpoint,
            max_display_width=max_display_width,
            grid_sample_size=grid_sample_size,
        )
        print(f"[AutoScan] Saved → {self.false_json}")

    def label(self):
        self._require_workspace()
        from .training.labeling import add_label_to_data, combine_and_shuffle

        print("[AutoScan] Labelling datapoints …")
        add_label_to_data(str(self.true_json), label=1, output_path=str(self.labeled_true_json))
        add_label_to_data(str(self.false_json), label=0, output_path=str(self.labeled_false_json))
        combine_and_shuffle(
            str(self.labeled_true_json),
            str(self.labeled_false_json),
            output_file=str(self.final_json),
        )
        print(f"[AutoScan] Saved → {self.final_json}")

    def train(self, epochs=100, batch_size=32, test_size=0.2, patience=10):
        self._require_workspace()
        if not self.final_json.exists():
            raise FileNotFoundError(f"Run label() first — {self.final_json} not found.")

        from .training.model import train as train_model

        with self.final_json.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        print("[AutoScan] Training model …")
        train_model(
            data,
            epochs=epochs,
            batch_size=batch_size,
            test_size=test_size,
            patience=patience,
            output_path=self.model_path,
        )
        print(f"[AutoScan] Model saved → {self.model_path}")

    def load_model(self):
        self._require_workspace()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.load_model_from_path(self.model_path)

    def load_model_from_path(self, path: str | Path):
        from tensorflow import keras

        model_path = Path(path).expanduser().resolve()
        self._model = keras.models.load_model(str(model_path), compile=False)
        print(f"[AutoScan] Model loaded ← {model_path}")

    def test(self, image_path, ratio=14, batch_size=4096, radius=5, thickness=-1):
        """Run the loaded detector on a path or a BGR NumPy image."""
        if self._model is None:
            raise RuntimeError("Load a model first with load_model() or load_model_from_path().")

        from .detector import test_grid_batched

        print("[AutoScan] Running inference ...")
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
