import unittest

import numpy as np

from flake_searcher.detector import test_grid_batched as run_detector


class _PredictableModel:
    def predict(self, features, batch_size, verbose):
        predictions = np.zeros((len(features), 3), dtype=np.float32)
        predictions[:, 0] = 1.0
        predictions[:5, 0] = 0.0
        predictions[:5, 1] = 1.0
        return predictions


class DetectorTests(unittest.TestCase):
    def test_grid_geometry_filtering_and_statistics_are_preserved(self):
        image = np.full((56, 70, 3), (20, 40, 60), dtype=np.uint8)
        matrix, display, stats = run_detector(
            image,
            _PredictableModel(),
            ratio=14,
            batch_size=4096,
            radius=2,
            thickness=-1,
        )

        expected = np.zeros((4, 5), dtype=np.uint8)
        expected[0, :] = 1
        np.testing.assert_array_equal(matrix, expected)
        self.assertEqual(stats["h"], 56)
        self.assertEqual(stats["w"], 70)
        self.assertEqual(stats["rows"], 4)
        self.assertEqual(stats["cols"], 5)
        self.assertEqual(stats["bg"], [60, 40, 20])
        self.assertEqual(stats["raw"], 5)
        self.assertEqual(stats["filtered"], 5)
        self.assertEqual(display.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
