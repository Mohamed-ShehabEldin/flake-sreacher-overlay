import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from flake_searcher.detector import test_grid_batched as run_detector
from flake_searcher.evaluation import (
    EvaluationError,
    PredictionRecorder,
    evaluate_image,
    hash_array,
    prepare_evaluation_paths,
    run_evaluation,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _DeterministicModel:
    def predict(self, features, batch_size, verbose):
        predictions = np.zeros((len(features), 3), dtype=np.float32)
        predictions[:, 0] = 1.0
        predictions[:5, 0] = 0.0
        predictions[:5, 1] = 1.0
        return predictions


class _ClassTwoModel:
    def predict(self, features, batch_size, verbose):
        predictions = np.zeros((len(features), 3), dtype=np.float32)
        predictions[:, 2] = 1.0
        return predictions


class EvaluationTests(unittest.TestCase):
    def _write_image(self, path):
        image = np.full((56, 70, 3), (20, 40, 60), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), image))
        return image

    def test_prediction_recorder_matches_direct_detector_output(self):
        image = np.full((56, 70, 3), (20, 40, 60), dtype=np.uint8)
        direct_mask, direct_overlay, direct_stats = run_detector(
            image, _DeterministicModel(), ratio=14, batch_size=4096, radius=2
        )
        recorded_mask, recorded_overlay, recorded_stats = run_detector(
            image, PredictionRecorder(_DeterministicModel()), ratio=14, batch_size=4096, radius=2
        )

        np.testing.assert_array_equal(recorded_mask, direct_mask)
        np.testing.assert_array_equal(recorded_overlay, direct_overlay)
        self.assertEqual(recorded_stats["raw"], direct_stats["raw"])
        self.assertEqual(recorded_stats["filtered"], direct_stats["filtered"])

    def test_repeated_outputs_have_one_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "input.png"
            self._write_image(image_path)
            result, _, _ = evaluate_image(
                image_path,
                PredictionRecorder(_DeterministicModel()),
                ratio=14,
                batch_size=4096,
                radius=2,
                repeats=3,
            )

        self.assertTrue(result["repeatable"])
        self.assertEqual(len(result["unique_output_hashes"]), 1)
        self.assertEqual(len({item["mask_hash"] for item in result["repeats"]}), 1)

    def test_output_must_not_overlap_an_input_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            input_dir = Path(folder) / "inputs"
            input_dir.mkdir()
            self._write_image(input_dir / "input.png")
            with self.assertRaisesRegex(EvaluationError, "must be separate"):
                prepare_evaluation_paths([input_dir], input_dir / "results")

    def test_output_must_not_enter_named_protected_repository_data(self):
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "input.png"
            self._write_image(image_path)
            protected_output = PROJECT_ROOT / "flakes" / "evaluation-results"
            with self.assertRaisesRegex(EvaluationError, "inside protected data"):
                prepare_evaluation_paths([image_path], protected_output)
            self.assertFalse(protected_output.exists())

    def test_run_preserves_input_and_writes_only_to_separate_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            input_dir = root / "inputs"
            output_dir = root / "results"
            input_dir.mkdir()
            image_path = input_dir / "input.png"
            self._write_image(image_path)
            model_path = root / "model.h5"
            model_path.write_bytes(b"synthetic model identity")
            before_files = sorted(input_dir.iterdir())
            before_hash = sha256_file(image_path)

            report = run_evaluation(
                model_path=model_path,
                input_paths=[input_dir],
                output_dir=output_dir,
                ratio=14,
                batch_size=4096,
                radius=2,
                repeats=3,
                save_masks=True,
                save_overlays=True,
                model_loader=lambda _: _DeterministicModel(),
            )

            self.assertEqual(sha256_file(image_path), before_hash)
            self.assertEqual(sorted(input_dir.iterdir()), before_files)
            self.assertTrue(report["input_integrity"]["verified"])
            self.assertTrue(report["repeatability"]["all_images_repeatable"])
            self.assertFalse(report["quality_metrics"]["available"])
            for metric in (
                "accuracy",
                "precision",
                "recall",
                "false_positive_rate",
                "miss_rate",
            ):
                self.assertIsNone(report["quality_metrics"][metric])
            self.assertTrue((output_dir / "report.json").is_file())
            self.assertTrue((output_dir / "per_image.csv").is_file())
            self.assertTrue((output_dir / "repeats.csv").is_file())
            self.assertEqual(len(list((output_dir / "masks").glob("*.png"))), 1)
            self.assertEqual(len(list((output_dir / "overlays").glob("*.png"))), 1)
            self.assertFalse(any(path.parent == input_dir for path in output_dir.rglob("*")))

    def test_class_two_observation_preserves_control_behavior(self):
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "input.png"
            self._write_image(image_path)
            result, mask, _ = evaluate_image(
                image_path,
                PredictionRecorder(_ClassTwoModel()),
                ratio=14,
                batch_size=4096,
                radius=2,
                repeats=1,
            )

        first = result["repeats"][0]
        self.assertEqual(first["raw_class_counts"], {"0": 0, "1": 0, "2": 20})
        self.assertEqual(first["raw_class_1_points"], 0)
        self.assertEqual(first["filtered_points"], 20)
        self.assertTrue(np.all(mask == 1))

    def test_control_manifest_pins_unchanged_detector_and_models(self):
        control = json.loads(
            (PROJECT_ROOT / "evaluation" / "control-v0.2.0.json").read_text(encoding="utf-8")
        )
        self.assertEqual(control["commit"], "ce830f7ba07dc4a8d3789edb0cc3b6c3d87e438a")
        self.assertEqual(
            sha256_file(PROJECT_ROOT / control["detector"]["path"]),
            control["detector"]["sha256"],
        )
        for filename, expected_hash in control["models"].items():
            with self.subTest(model=filename):
                actual = hashlib.sha256(
                    (PROJECT_ROOT / "models" / "detector" / filename).read_bytes()
                ).hexdigest()
                self.assertEqual(actual, expected_hash)

    def test_schemas_are_definitions_not_placeholder_annotations(self):
        report_schema = json.loads(
            (PROJECT_ROOT / "evaluation" / "report.schema.json").read_text(encoding="utf-8")
        )
        annotation_schema = json.loads(
            (PROJECT_ROOT / "evaluation" / "annotation.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report_schema["properties"]["control"]["const"], "v0.2.0")
        self.assertIn("complete_review", annotation_schema["properties"]["images"]["items"]["required"])
        self.assertFalse((PROJECT_ROOT / "evaluation" / "annotations.json").exists())



if __name__ == "__main__":
    unittest.main()
